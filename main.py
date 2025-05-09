import yaml

from scripts import Task

def main() -> None:
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    task = Task(config)
    task.run()

if __name__=='__main__':
    main()