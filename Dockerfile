FROM ubuntu

WORKDIR /workspaces/underwater

RUN apt-get update
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3.8 python3-pip libsndfile1 libsndfile1-dev nano ffmpeg libavcodec-extra

# Install the requirements for the Underwater Sound Classification.
ADD requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Install the development requirements for the Underwater Sound Classification.
ADD requirements-dev.txt .
RUN pip install -r requirements-dev.txt
RUN rm requirements-dev.txt

# Set Python 3 as default for python command.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Make the terminal colored.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc
