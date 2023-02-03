import docker
from rich.progress import Progress

tasks = {}

def docker_image_pull_with_progress(client, image_name):

    # Show task progress (red for download, green for extract)
    def show_progress(line, progress):
        if line['status'] == 'Downloading':
            id = f'[red][Download {line["id"]}]'
        elif line['status'] == 'Extracting':
            id = f'[green][Extract  {line["id"]}]'
        else:
            # skip other statuses
            return

        if id not in tasks.keys():
            tasks[id] = progress.add_task(f"{id}", total=line['progressDetail']['total'])
        else:
            progress.update(tasks[id], completed=line['progressDetail']['current'])


    print(f'Pulling image: {image_name}')
    with Progress() as progress:
        resp = client.api.pull(image_name, stream=True, decode=True)
        for line in resp:
            show_progress(line, progress)

def docker_container_run(
    client,                 # docker.from_env()
    image,                  # image name
    command,                # command to run in container
    stdout=False,           # disable logs
    stderr=False,           # disable stderr logs
    detach=True,            # detached mode - daemon
    remove=True,            # remove fs after exit
    auto_remove=True,       # remove fs if container fails
    device_requests=None,   # pass GPU
    network_mode='host',    # --net=host
    volumes=[]              # -v
):
    ctainer = None
    try:
        ctainer = client.containers.run(
            image=image,
            command=command,
            stdout=stdout,
            stderr=stderr,
            detach=detach,
            remove=remove,
            auto_remove=auto_remove,
            runtime='nvidia',   # Use nvidia-container-runtime
            device_requests=device_requests,
            network_mode=network_mode,
            volumes=volumes
        )
    except Exception as ex:
        print('Start model-server with NVIDIA runtime failed.')
        print('Trying without NVIDIA runtime')

        try:
            ctainer = client.containers.run(
                image=image,
                command=command,
                stdout=stdout,
                stderr=stderr,
                detach=detach,
                remove=remove,
                auto_remove=auto_remove,
                device_requests=device_requests,
                network_mode=network_mode,
                volumes=volumes
            )
        except Exception as ex:
            print('ERROR: Unable to start model server:')
            print(f'ERROR: {ex}')

    print('Started model server successfully')
    return ctainer

if __name__ == '__main__':
    # Pull a large image
    client = docker.from_env()
    IMAGE_NAME = 'bitnami/pytorch'
    docker_image_pull_with_progress(client, IMAGE_NAME)
