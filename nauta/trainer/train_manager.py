import math
from tqdm import tqdm

from nauta.tools.utils import plot_confusion_matrix


class TrainManager:
    def __init__(
        self, model, loss_fn, optimizer, lr_scheduler, train_dataloader, validation_dataloader,
        epochs, initial_epoch=0, metrics={}, reference_metric="", writer=None, device='cpu',
        early_stop=True,
        ):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.writer = writer
        self.metrics = metrics
        self.reference_metric = reference_metric
        self.epochs = epochs
        self.initial_epoch = initial_epoch

        self.best_measure = 0

        self.current_validation_loss = 0
        self.last_validation_loss = 0

        self.early_stop = early_stop
        self.trigger_times = 0
        self.patience = 4
        return

    def _efficient_zero_grad(self, model):
        for param in model.parameters():
            param.grad = None

    def _train_single_epoch(self, epoch):
        self.model.train()
        step = epoch * len(self.train_dataloader)
        train_loss = 0
        for input_data, target_data in tqdm(self.train_dataloader,
            desc=f"Train", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            ):

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # calculate loss
            self._efficient_zero_grad(self.model)
            prediction = self.model(input_data)
            loss = self.loss_fn(prediction, target_data)
            train_loss += loss.item()

            step += 1
            self.writer.add_scalar('Loss/train', loss, step)

            # backpropagate error and update weights
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss/len(self.train_dataloader)

        self.writer.add_scalar('Loss/train_epoch', train_loss, step)
        print(f"Loss: {train_loss:.4f}")
        return train_loss

    def _validate_single_epoch(self, epoch):
        self.model.eval()
        self.last_validation_loss = self.current_validation_loss
        self.current_validation_loss = 0
        display_values = []
        for input_data, target_data in tqdm(self.validation_dataloader,
            desc=f"Validation", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            ):

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            prediction = self.model(input_data)

            loss = self.loss_fn(prediction, target_data)
            self.current_validation_loss += loss.item()

            for metric in self.metrics:
                self.metrics[metric](prediction, target_data)

        use_reference=False
        if self.reference_metric in self.metrics:
            use_reference=True

        self.current_validation_loss = self.current_validation_loss/len(self.validation_dataloader)
        self.writer.add_scalar(f'Loss/validation', self.current_validation_loss, epoch)
        display_values.append(f"Loss: {self.current_validation_loss:.4f}")

        for idx, metric in enumerate(self.metrics):
            value = self.metrics[metric].compute()
            if idx == 0:
                ref_metric = value
            if use_reference:
                if metric == self.reference_metric:
                    ref_metric = value
            if metric == "ConfusionMatrix":
                cm_fig = plot_confusion_matrix(
                    value.cpu().detach().numpy(), class_names=self.validation_dataloader.dataset.class_mapping.keys()
                )
                self.writer.add_figure(f'Metrics/{metric}', cm_fig, epoch)
            else:
                display_values.append(f"{metric}: {value:.4f}")
                self.writer.add_scalar(f'Metrics/{metric}', value, epoch)
            self.metrics[metric].reset()

        print("  ".join(display_values))

        return ref_metric

    def start_train(self, checkpoint_manager=None):
        for epoch in range(self.initial_epoch, self.epochs):
            print(f"Epoch {epoch+1}")
            loss = self._train_single_epoch(epoch)
            measure = self._validate_single_epoch(epoch)

            self.writer.add_scalar(f'Hyper/lr', self.optimizer.param_groups[0]["lr"], epoch)
            self.lr_scheduler.step()

            if measure.cpu().detach().numpy() > self.best_measure:
                self.best_measure = measure.cpu().detach().numpy()

            # Save a checkpoint.
            if checkpoint_manager is not None:
                checkpoint_manager.save(epoch, measure=math.floor(self.best_measure * 1000000))

            print("---------------------------")

            # Early stopping
            if self.early_stop:
                if self.current_validation_loss > self.last_validation_loss:
                    self.trigger_times += 1
                    if self.trigger_times >= self.patience:
                        print('Early stopping!\n')
                        break
                else:
                    self.trigger_times = 0

        print("Finished training")
