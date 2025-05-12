import yaml

from scripts import Task

def main() -> None:
    """
    The main script of the project. 
    It reads the config file and runs the code.  
    """
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    task = Task(config)
    task.run()

if __name__=='__main__':
    main()