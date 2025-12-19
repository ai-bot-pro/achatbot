# --------
# build base img
docker_cpu_debian_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:base \
		--build-arg A_PYTHON_IMG_TAG=3.11-slim \
		--build-arg A_ACHATBOT_DEPENDENCY=achatbot \
		--no-cache \
		-f Dockerfile.cpu.debian .

# build fastapi_daily_bot img
docker_cpu_debian_fastapi_daily_bot_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:fastapi_daily_bot \
		--build-arg A_PYTHON_IMG_TAG=3.11-slim \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server,daily_rtvi_bot,daily_langchain_rag_bot,speech_vad_analyzer,deepgram_asr_processor,openai,tts_edge]" \
		--no-cache \
		-f Dockerfile.cpu.debian .

# build lam_audio2expression_avatar img
docker_cpu_debian_lam_audio2expression_avatar_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:lam_audio2expression_avatar \
		--build-arg A_PYTHON_IMG_TAG=3.11-slim \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server,silero_vad_analyzer,sense_voice_asr,openai_llm_processor,google_llm_processor,litellm_processor,together_ai,tts_edge,lam_audio2expression_avatar]" \
		--no-cache \
		-f Dockerfile.cpu.debian .


# build runnable img
# install all vad,asr,llm,tts dependency, need achatbot:base img
docker_cpu_debian_fastapi_daily_bot_run_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:fastapi_daily_bot_run_all \
		--build-arg A_ACHATBOT_IMG_TAG=base \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, asr_processor, llm_processor, tts_processor]" \
		--no-cache \
		-f fastapi-daily-chat-bot/Dockerfile.cpu.debian .

# install vad,api asr,api llm,api tts dependency, need achatbot:base img
docker_cpu_debian_fastapi_daily_bot_run_api_models_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:fastapi_daily_bot_run \
		--build-arg A_ACHATBOT_IMG_TAG=base \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server, daily_rtvi_bot, daily_langchain_rag_bot, speech_vad_analyzer, deepgram_asr_processor, openai, tts_edge]" \
		--no-cache \
		-f fastapi-daily-chat-bot/Dockerfile.cpu.debian .

# install lam_audio2expression_avatar dependency, need achatbot:base img
docker_cpu_debian_lam_audio2expression_avatar_run_img:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker build \
		-t achatbot:lam_audio2expression_avatar_run \
		--build-arg A_ACHATBOT_IMG_TAG=base \
		--build-arg A_ACHATBOT_DEPENDENCY="achatbot[fastapi_bot_server,silero_vad_analyzer,sense_voice_asr,openai_llm_processor,google_llm_processor,litellm_processor,together_ai,tts_edge,lam_audio2expression_avatar]" \
		--no-cache \
		-f avatar-chat-bot/Dockerfile.lam.cpu.debian .

#---- run container with runnable img ----

docker_cpu_debian_fastapi_daily_bot_container_run:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker run \
		-it \
		-p 8000:8000 \
		--name achatbot_fastapi_daily_bot \
		--env-file ../../.env \
		achatbot:fastapi_daily_bot_run

docker_cpu_debian_fastapi_daily_bot_containers_compose_run:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker compose \
		-f fastapi-daily-chat-bot/docker-compose.yaml up

docker_cpu_debian_lam_audio2expression_avatar_container_run:
	DOCKER_HOST=unix://$(HOME)/.docker/run/docker.sock docker run \
		-it \
		-p 4321:4321 \
		-v ../../config:/root/.achatbot/config \
		-v ../../models:/root/.achatbot/models \
		--name achatbot_lam_audio2expression_avatar \
		--env-file ../../.env \
		achatbot:lam_audio2expression_avatar_run

#-------tag for push to docker hub-------
docker_cpu_debian_base_tag:
	DOCKER_HOST=unix://$HOME/.docker/run/docker.sock docker tag \
		achatbot:base weedge/achatbot:base

docker_cpu_debian_fastapi_daily_bot_run_tag:
	DOCKER_HOST=unix://$HOME/.docker/run/docker.sock docker tag \
		achatbot:fastapi_daily_bot_run weedge/achatbot:fastapi_daily_bot_run

#----- publish -----
docker_cpu_debian_base_tag_push:
	DOCKER_HOST=unix://$HOME/.docker/run/docker.sock docker push \
		weedge/achatbot:base

docker_cpu_debian_fastapi_daily_bot_tag_push:
	DOCKER_HOST=unix://$HOME/.docker/run/docker.sock docker push \
		weedge/achatbot:fastapi_daily_bot_run