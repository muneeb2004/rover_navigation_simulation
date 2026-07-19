# Project 8: Rover Navigation Simulator

A planetary rover simulation that computes position, heading, and trajectories on Windows using linear algebra transformation matrices.

## Group Members
* **Sheikh M Muneeb**
* **Misha Jessani**
* **Adil Saleem**

---

## Project Requirements & Implementation Summary

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **1. Represent rover position as a vector** | Position $[x, y]^T$, state vector $[x, y, \theta]^T$, and homogeneous vector $[x, y, 1]^T$ | Verified |
| **2. Implement movement operations** | Forward & backward translation vectors and transformation matrices $T(t_x, t_y)$ | Verified |
| **3. Apply rotation matrices** | 2D rotation matrices $R(\theta)$ and combined transformation matrices $T(\theta, t_x, t_y)$ | Verified |
| **4. Chain multiple transformations** | Order-dependent matrix multiplication $W_n = W_0 \cdot T_1 \cdot T_2 \cdots T_n$ | Verified |
| **5. Display rover's trajectory** | Matplotlib 2D trajectory rendering with heading arrows, path history, and PNG export | Verified |

### Mathematical Concepts Explored
* **Matrix multiplication**: Chronological composition of local command matrices $W_n = W_{n-1} \cdot T_{\text{local}}$.
* **Rotation matrices**: 2D rotation $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$.
* **Translation**: Displacement vectors $\mathbf{t} = [t_x, t_y]^T$.
* **Coordinate transformations**: Local rover frame vs. global world frame.
* **Homogeneous coordinates**: Unifying rotation and translation in a $3\times3$ matrix.

---

## System Requirements & Setup

The project requires Python 3.8+ along with the dependencies listed in [requirements.txt](file:///D:/Habib%20Uni/Summer%20Sem/Project/requirements.txt):
* **NumPy** for matrix transformations
* **Matplotlib** for plotting the rover's trajectory

Install dependencies using `pip`:
```cmd
pip install -r requirements.txt
```

---

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
