# Rover Navigation Simulator

A planetary rover simulation that computes position, heading, and trajectories on Windows.

## Requirements

The project requires Python 3.8+ along with the dependencies listed in [requirements.txt](file:///D:/Habib%20Uni/Summer%20Sem/Project/requirements.txt):
* **NumPy** for matrix transformations
* **Matplotlib** for plotting the rover's trajectory

Install dependencies using `pip`:
```cmd
pip install -r requirements.txt
```

## Running the Program on Windows

We have provided two convenient launcher scripts to execute the simulator:

### 1. Command Prompt / Double-Click launcher: [run.bat](file:///D:/Habib%20Uni/Summer%20Sem/Project/run.bat)
You can run it from the command line:
```cmd
run.bat
```
Or with specific options:
* Run tests: `run.bat --run-tests`
* Run without showing GUI: `run.bat --no-show`
* Run using a custom commands file: `run.bat --commands-file path\to\commands.txt`
* Save the plot to an image: `run.bat --save-plot path\to\plot.png`

*(If you double-click `run.bat` in Windows Explorer, the terminal window will remain open after execution so you can review the console outputs.)*

### 2. PowerShell launcher: [run.ps1](file:///D:/Habib%20Uni/Summer%20Sem/Project/run.ps1)
You can run it in PowerShell:
```powershell
.\run.ps1
```
Or with parameters:
```powershell
.\run.ps1 --run-tests
```
# rover_navigation_simulation
