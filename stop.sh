# Adapted from https://github.com/jkarolczak/ligand-classification
if [[ $1 == "cpu" ]]; then
    docker compose --env-file docker/.env -f docker/compose-cpu.yml down
elif [[ $1 == "deploy" ]]; then
    docker compose --env-file docker/.env -f docker/compose-deploy.yml down
else
    docker compose --env-file docker/.env -f docker/compose-gpu.yml down
fi
