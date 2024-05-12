# ROV Red Square Detection and Navigation

<div style="text-align:center">
  <img src="./Autonomous_Task.gif" alt="Autonomous Task GIF">
</div>

## Overview
This repository contains autonomous software developed for a Remotely Operated Vehicle (ROV) project. The software utilizes image processing techniques to detect 15x15 red squares in the ROV's environment and provides real-time feedback for precise navigation in underwater conditions.

## Contributors
- [Fares Hazem](https://www.linkedin.com/in/fares-hazem-b5590214b/)
- [Hatem Mohamed](https://www.linkedin.com/in/hatem-mohamed-6175b0244/)
- [Ahmed Ehab](https://www.linkedin.com/in/ahmed-ehab-491a39233/)

## Project Structure
- **Code:** Contains the actual software for red square detection and navigation.
  - **Decision_Making_Block:** Module for decision-making algorithms.
  - **Detection_and_Preprocessing_Block:** Module for detection and preprocessing functionalities.
  - **main.py:** Main script for executing the detection and navigation task.
  - **tester.py:** Script for testing and validation purposes.
- **Cocktail_Scripts:** Includes scripts for tuning and testing purposes.

### Description of Code Contents
- **Decision_Making_Block:**
  - decisionMaking: Module for decision-making algorithms.
  - distanceEstimator: Module for estimating distances.
- **Detection_and_Preprocessing_Block:**
  - DetectionPreprocessing: Module for detection and preprocessing.
  - framePreprocessor: Module for preprocessing frames.
  - yoloDetector: Module for detecting objects using YOLO algorithm.

## Usage
1. Clone the repository to your local machine.
2. Navigate to the 'Code' folder.
3. Configure parameters in 'main.py' as needed.
4. Run 'main.py' to initiate the detection and navigation task.
