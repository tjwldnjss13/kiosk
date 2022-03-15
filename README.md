# kiosk

Real-time kiosk user analysis system using pedestrian/face detection and age/gender classification

## Environment

### Nvidia Jetson Nano

  ![JetsonNano-DevKit_Front-Top_Right_trimmed](https://user-images.githubusercontent.com/48514976/158322188-9f75a34d-d678-454a-9cc4-e4a617812abf.jpg)
  
  (Source : https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
  
### Model
  
![model](https://user-images.githubusercontent.com/48514976/158328202-a90b2bfc-6cc1-4118-b44d-c9fd1ea3fbe2.JPG)

  
#### Detection

Pedestrian and face detection network using MobileNet and skip connection

#### Classification

Age and gender classification network using Mobilenet and CBAM

![age gender](https://user-images.githubusercontent.com/48514976/158328359-36952a8c-1d53-4a20-beff-8b73f31c025e.JPG)
  
Age is predicted in 9 super categories and 9 set of 5-way sub categories.
  
![age classification](https://user-images.githubusercontent.com/48514976/158328376-c7409aa8-b67c-4c33-9e29-1e94459f101f.JPG)

