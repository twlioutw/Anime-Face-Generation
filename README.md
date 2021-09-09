# anime-face-generatrion
Using GAN to generate anime faces of given hair and eye color.

## Requirements

Python 3.7

Keras==2.3.1

tensorflow==2.1.0


<!-- <p align="center">
  <img src="https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/5000.jpg" width="24%" />
  <img src="https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/10000.jpg" width="24%"  /> 
  <img src="https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/50000.jpg" width="24%"/>
  <img src="https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/100000.jpg" width="24%"/>
</p>
 -->

## Results

![](https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/5000.jpg) **5000 iterations** |  ![](https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/10000.jpg) **10000 iterations** | ![](https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/50000.jpg) **50000 iterations** | ![](https://github.com/twlioutw/Anime-Face-Generation/blob/main/images/100000.jpg) **100000 iterations**
:--:|:--:|:--:|:--:
<!--  **5000 iterations**| **10000 iterations** | **50000 iterations** | **100000 iterations** -->


## Usage
To generate an anime face image:
```
python generate.py [--hair color] [--eye color] [output path]
```


## Available color
**hair:** orange, white, aqua, gray, green, red, purple, pink, blue, black, brown, blonde

**eye:** gray, black, orange, pink, yellow, aqua, purple, green, brown, red, blue
