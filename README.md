# Combatant Object Detection in Rust(in Squad)

## Introduction

This API can detect combatants in the military simulation game 'Squad'. The goal of this project is to create a method of using AI of sorts, to fulfil the duties of an FPV drone pilot.

This project was built using Rocket in Rust, with some HTML to be rendered to the page. The model was trained using the Ultralytics repo and YOLOv8, trained on a custom dataset.

The intended usage of the program is to submit an image, and have the custom-trained object detection model look for combatants. These combatants are highlighted and the combatant that is closest to the centre will be selected and targeted by the drone. A vector to the centre of the drones vision to the centre of the target is created and the drone should maneuver as is necessary to steer towards the centre of the target.

![Four Russians detected with more than 90% accuracy](https://github.com/FionnCL/fpv-computer-vision-api/blob/main/github-markdown-image-1.png?raw=true)

## Usage

To run the program make sure you have Rust and Cargo installed.

Once this is done, input the command `cargo run` into your terminal. The website URL will be displayed in the terminal once you input the command. It should be available at: [127.0.0.0](http://127.0.0.1:8000/).

Once on the page, submit an image using the prompt and all detected combatants will be highlighted in the image.

## Discussion

This program is more of a proof of concept, as I am not going to be figuring out how to interface the code seen here with phyiscal drones.

Displayed on the page(when an image is submitted), will be information that would theoretically be used to by a drone to steer into it's target. The program outputs the distance from the centre of the screen to the centre of the target in pixels. With the centre of the image being (0,0). This data was left so raw as how the drone would wish to interact with these figures would be undecided, however, leaving it this open to interpretation does make it so that it would be easy to imagine how a drone MIGHT interact with this information.

The information was given as it is because I theorize that by giving the vector from the centre of the screen to the centre of the target in the vector's individual i and j components that are relative to the size of the image, you should be able to smoothly AND efficiently steer the drone(when targeting).

## Final Notes

This was trained using an extremely small dataset of soliders from the incredibly realistic game 'Squad'. The API is quite slow all things considered. This could easily be fixed by utilizing the CUDA toolkit to run on your GPU, I did not do this however and as seen in the license, you are free to do so if you wish.

This is quite a hastily put-together project which I started with no prior knowledge of Rust or Computer Vision, so pardon my code. If there are any changes you'd want to suggest, feel free to make the change and push request!
