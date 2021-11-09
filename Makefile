include .env

USER=thomasj27
REPO=mirror_${DEVICE}_${VERSION}
IMNAME = ${USER}/${REPO}

delete:
	nvidia-docker image rm -f $(IMNAME)

clear:
	nvidia-docker rm $(REPO)

build:
	nvidia-docker build -t $(IMNAME) -f build/${DEVICE}/Dockerfile .

push:
	nvidia-docker push ${IMNAME}

pull:
	git pull
	-git lfs fetch --all

run:
	-nvidia-docker rm $(REPO)
	nvidia-docker run -d --expose 5000 -e PYTHONUNBUFFERED=1 --network="host" --privileged --volume=/dev:/dev -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=${DISPLAY} -e QT_X11_NO_MITSHM=1 --name=$(REPO) $(IMNAME)

launch:
ifeq (${DEVICE}, gpu)
	-nvidia-docker rm $(REPO)
	nvidia-docker run --expose 5000 -e PYTHONUNBUFFERED=1 --network="host" --privileged --volume=/dev:/dev -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=${DISPLAY} -e QT_X11_NO_MITSHM=1 --name=$(REPO) $(IMNAME)
else
	-nvidia-docker rm $(REPO)
	nvidia-docker run --expose 5000 -e PYTHONUNBUFFERED=1 --network="host" --privileged --volume=/dev:/dev --name=$(REPO) $(IMNAME)
endif

stop:
	docker stop -t 0 $(REPO)

open:
	chromium http://127.0.0.1:8000 --start-fullscreen --disk-cache-dir=/dev/null --disk-cache-size=1 --media-cache-size=1 --incognito

open_ssh:
	DISPLAY=:0 chromium http://127.0.0.1:8000 --start-fullscreen --disk-cache-dir=/dev/null --disk-cache-size=1 --media-cache-size=1 --incognito

deploy:
	cp -r deployement /home/$USER/.config/systemd/user/

server:
	python3 -m http.server --directory app/

logs:
	docker logs --follow $(REPO)

service_stop:
	systemctl --user stop mirror.service
	systemctl --user stop mirrorfront.service
	systemctl --user stop mirrorstartchrome.service

service_start:
	systemctl --user start mirror.service
	systemctl --user start mirrorfront.service
	systemctl --user start mirrorstartchrome.service

waiting:
	DISPLAY=:0 chromium apps/waiting/index.html --start-fullscreen --incognito
