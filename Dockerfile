FROM python:3.6

ADD . /pommerman-docker

RUN pip3 install --no-cache-dir -r pommerman-docker/requirements.txt

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /pommerman-docker
ENTRYPOINT ["python3"]
CMD ["run.py"]