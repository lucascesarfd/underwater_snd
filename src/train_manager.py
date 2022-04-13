
from tqdm import tqdm
from utils import plot_confusion_matrix


class TrainManager:
    def __init__(
        self, model, loss_fn, optimizer, lr_scheduler, train_dataloader, validation_dataloader,
        epochs, initial_epoch=0, metrics={}, reference_metric="", writer=None, device='cpu'
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
        return

    def _efficient_zero_grad(self, model):
        for param in model.parameters():
            param.grad = None

    def _train_single_epoch(self, epoch):
        self.model.train()
        step = epoch * len(self.train_dataloader)
        for input_data, target_data in tqdm(self.train_dataloader):

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # calculate loss
            self._efficient_zero_grad(self.model)
            prediction = self.model(input_data)
            loss = self.loss_fn(prediction, target_data)

            step += 1
            self.writer.add_scalar('Loss/train', loss, step)

            # backpropagate error and update weights
            #self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"Train Loss: {loss.item()}")
        return loss.item()

    def _validate_single_epoch(self, epoch):
        self.model.eval()
        for input_data, target_data in tqdm(self.validation_dataloader):

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            prediction = self.model(input_data)
            for metric in self.metrics:
                self.metrics[metric](prediction, target_data)

        use_reference=False
        if self.reference_metric in self.metrics:
            use_reference=True

        for idx, metric in enumerate(self.metrics):
            value = self.metrics[metric].compute()
            if idx == 0:
                ref_metric = value
            if use_reference:
                if metric == self.reference_metric:
                    ref_metric = value
            if metric == "ConfusionMatrix":
                cm_fig = plot_confusion_matrix(
                    value.numpy(), class_names=self.validation_dataloader.dataset.class_mapping.keys()
                )
                self.writer.add_figure(f'Metrics/{metric}', cm_fig, epoch)
            else:
                print(f"Validation {metric}: {value}")
                self.writer.add_scalar(f'Metrics/{metric}', value, epoch)
            self.metrics[metric].reset()

        return ref_metric

    def start_train(self, checkpoint_manager=None):
        for epoch in range(self.initial_epoch, self.epochs):
            print(f"Epoch {epoch+1}")
            loss = self._train_single_epoch(epoch)
            measure = self._validate_single_epoch(epoch)
            self.lr_scheduler.step()

            is_best = False
            if measure.cpu().detach().numpy() > self.best_measure:
                is_best = True
                self.best_measure = measure.cpu().detach().numpy()

            # Save a checkpoint.
            if checkpoint_manager is not None:
                checkpoint_manager.save(epoch, is_best=is_best)

            print("---------------------------")
        print("Finished training")
