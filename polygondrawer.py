# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:47:59 2020

@author: labadmin

taken from: https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python/37235130

"""

import numpy as np
import cv2


class PolygonDrawer(object):

    FINAL_LINE_COLOR = (255, 0, 255)# (255, 255, 255)
    BG_COLOR = (0, 255, 0)
    
    """
    Interactive polygon-drawing GUI for identifying cell body and neuropil
    INPUTS:
            window_name: name of window (str)
            inputImage: max projection image of cell
            neuropilThickness: pixel thickness of neuropil around cell
            neuropilOffset: pixel offset from neuropil
            
    RETURNS:
            canvas: cell mask
            canvas_neuropil: neuropil mask
    """
    def __init__(self, window_name, inputImage, neuropilThickness=10, neuropilOffset=3):
        self.window_name = window_name # Name for our window
        self.CANVAS_SIZE = np.shape(inputImage)
        
        image = np.repeat(inputImage[:,:,np.newaxis], 3, axis=2)
        self.image = image
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        
        self.neuropilThickness = neuropilThickness
        self.neuropilOffset = neuropilOffset
        
    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        # cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas = self.image
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, self.FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                # cv2.line(canvas, self.points[-1], self.current, self.WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing of a filled polygon
        canvas = np.zeros(self.CANVAS_SIZE, np.uint8)
        canvas_neuropil = np.zeros(self.CANVAS_SIZE, np.uint8)
        canvas_neuropil_offset = np.zeros(self.CANVAS_SIZE, np.uint8)
        
        if (len(self.points) > 0):
            cv2.fillPoly(canvas, np.array([self.points]), self.FINAL_LINE_COLOR)
            
            # cv2.fillPoly(canvas_neuropil, np.array([self.points]), self.FINAL_LINE_COLOR, offset=(20,20))
            cv2.polylines(canvas_neuropil, np.array([self.points]), True, self.FINAL_LINE_COLOR, thickness=self.neuropilThickness)
            cv2.polylines(canvas_neuropil_offset, np.array([self.points]), True, self.FINAL_LINE_COLOR, thickness=self.neuropilOffset) # offset 
            
            canvas_neuropil = canvas_neuropil - canvas - canvas_neuropil_offset
        # And show it
        #cv2.imshow(self.window_name, canvas)
        cv2.imshow(self.window_name, np.concatenate((canvas, canvas_neuropil), axis=0))
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyAllWindows()
        return (canvas, canvas_neuropil)
