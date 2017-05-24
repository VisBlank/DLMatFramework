'''
Execute a command for each element on a sub-directory

The fire framework will allow automatic creation of CLI (command line interface) from the source code
So for example
python execute_on_dir.py for_each data './darknet detect cfg/yolo.cfg yolo.weights'

This will execute the function for_each passing the parametes "data" and "darknet detect cfg/yolo.cfg yolo.weights"

References:
    https://stackoverflow.com/questions/1120707/using-python-to-execute-a-command-on-every-file-in-a-folder
    https://github.com/google/python-fire
    https://stackoverflow.com/questions/1685157/python-specify-popen-working-directory-via-argument
    https://github.com/google/python-fire
    https://stackoverflow.com/questions/21406887/subprocess-changing-directory
    
'''
import fire
import os
import subprocess
import fnmatch

class RunCommandForEach(object):

    # Execute something
    def execute_command(self, path, command):
        complete_command = command + ' ' + path
        print(complete_command)
        subprocess.call(complete_command, cwd=os.getcwd(), shell=True)

    # Iterate on directory folder executing something for each element
    def for_each(self, folder, command):
        # Declare image filters
        includes = ['*.jpg', '*.png']
        for directory, subdirectories, files in os.walk(folder, topdown=True):
            for file in files:
                # Create the full path
                file_path = os.path.join(directory, file)

                # For each filter
                for filter in includes:
                    # Check if filename match filter
                    if fnmatch.fnmatch(file, filter):
                        # Execute command
                        self.execute_command(file_path, command)


if __name__ == '__main__':
  # Execute fire with an object
  fire.Fire(RunCommandForEach)

