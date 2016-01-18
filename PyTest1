# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:33:05 2015

@author: dacohe
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

a=np.arange(1,11)
b=a*a
plt.figure()
plt.subplot(221)
plt.plot(a,b)

#plt.figure()
plt.subplot(222)
im=((1,1),(1,3))
plt.imshow(im)
plt.colorbar()

theta=np.linspace(0,2*np.pi,30)
y1=np.sin(theta)
y2=np.sin(1.5*theta)
#plt.figure()
plt.subplot(223)
plt.plot(theta, y1, 'ro-',linewidth=2, label='line y1')
plt.plot(theta, y2, 'g*-',linewidth=2, label='line y2')
plt.grid()
str1="Theta"
plt.xlabel(str1)
plt.ylim((-1.2,1.2))
plt.title("title 1")
plt.legend()


theta=np.arange(0,2*np.pi,0.1)
y1=np.sin(theta)
y2=np.sin(1.5*theta)
#plt.figure()
plt.subplot(224)
plt.plot(theta, y1, 'ro-',theta, y2, 'b^-',linewidth=2, label='line 99')
plt.grid()
str1="Theta"
plt.xlabel(str1)
plt.xlim((-1,10))
plt.title("title 2")
plt.legend()

plt.savefig('Example1.eps')

r=np.sin(2*theta)
plt.figure()
plt.polar(r,theta)

plt.savefig('Example2.eps')

img = cv2.imread('IMG_1810.jpg')
plt.figure(3)
plt.imshow(img)
