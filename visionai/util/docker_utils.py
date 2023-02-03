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

if __name__ == '__main__':
    # Pull a large image
    client = docker.from_env()
    IMAGE_NAME = 'bitnami/pytorch'
    docker_image_pull_with_progress(client, IMAGE_NAME)
