# ============================================================================
# Copyright (c) 2001-2019 FLIR Systems, Inc. All Rights Reserved.

# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# ============================================================================
#
# AcquisitionMultipleCamera.py shows how to capture images from
# multiple cameras simultaneously. It relies on information provided in the
# Enumeration, Acquisition, and NodeMapInfo examples.
#
# This example reads similarly to the Acquisition example,
# except that loops are used to allow for simultaneous acquisitions.

import os
import PySpin
import _thread
import multiprocessing

class AviType:
    """'Enum' to select AVI video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264 = 2

chosenAviType = AviType.H264  # change me!
NUM_IMAGES = 500  # number of images to grab
images = {}

def save_list_to_avi(nodemap, nodemap_tldevice, image):
    """
    This function prepares, saves, and cleans up an AVI video from a vector of images.

    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param image: List of images to save to an AVI video.
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type images: list of ImagePtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** CREATING VIDEO ***')

    try:
        result = True

        # Retrieve device serial number for filename
        device_serial_number = ''
        node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

        if PySpin.IsAvailable(node_serial) and PySpin.IsReadable(node_serial):
            device_serial_number = node_serial.GetValue()
            print('Device serial number retrieved as %s...' % device_serial_number)

        # Get the current frame rate; acquisition frame rate recorded in hertz

        node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(node_acquisition_framerate) and not PySpin.IsReadable(node_acquisition_framerate):
            print('Unable to retrieve frame rate. Aborting...')
            return False
        # node_acquisition_framerate

        framerate_to_set = node_acquisition_framerate.GetValue()

        print('Frame rate to be set to %d...' % framerate_to_set)

        # Select option and open AVI filetype with unique filename
        avi_recorder = PySpin.SpinVideo()

        if chosenAviType == AviType.UNCOMPRESSED:
            avi_filename = 'SaveToAvi-Uncompressed-%s' % device_serial_number

            option = PySpin.AVIOption()
            option.frameRate = framerate_to_set

        elif chosenAviType == AviType.MJPG:
            avi_filename = 'SaveToAvi-MJPG-%s' % device_serial_number

            option = PySpin.MJPGOption()
            option.frameRate = framerate_to_set
            option.quality = 30

        elif chosenAviType == AviType.H264:
            avi_filename = 'SaveToAvi-H264-%s' % device_serial_number

            option = PySpin.H264Option()
            option.frameRate = framerate_to_set
            option.bitrate = 5000000 #decides the level of compression
            option.height = image[0].GetHeight()
            print('image height = %d' % (option.height))
            option.width = image[0].GetWidth()
            print('image height = %d' % (option.width))

        else:
            print('Error: Unknown AviType. Aborting...')
            return False

        avi_recorder.Open(avi_filename, option)

        # Construct and save AVI video

        print('Appending %d images to AVI file: %s.avi...' % (len(image), avi_filename))

        print("Length of images is %d" %(len(image)))
        for i in range(len(image)):
            avi_recorder.Append(image[i])
            print('Appended image %d...' % i)

        # Close AVI file
        #
        avi_recorder.Close()
        print('Video saved at %s.avi' % avi_filename)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def acquire_image(i,n):
    try:
        result = True

        # acquire image
        image_result = cam_list[i].GetNextImage()

        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ... \n' % image_result.GetImageStatus())
        else:
            # Print image information
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            print('Camera %d grabbed image %d, width = %d, height = %d' % (i, n, width, height))

            # Convert image to BayerRG8
            images[i].append(image_result.Convert(PySpin.PixelFormat_BayerRG8, PySpin.HQ_LINEAR))
            #images.append(image_result)
            if n == 0:
                print("Saving first image")
                image_result.Save("Trial_image_of_cam_%d.jpg" %(i))


        # Release image
        image_result.Release()
        print()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def print_device_info(nodemap, cam_num):
    """
    This function prints the device information of the camera from the transport
    layer;
    :param nodemap: Transport layer device nodemap.
    :param cam_num: Camera number.
    :type nodemap: INodeMap
    :type cam_num: int
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print('Printing device information for camera %d... \n' % cam_num)

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print('%s: %s' % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

        else:
            print('Device control information not available.')
        print()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return result

def run_multiple_cameras():
    """
    :param cam_list: List of cameras
    :type cam_list: CameraList
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Retrieve transport layer nodemaps and print device information for
        # each camera
        print('*** DEVICE INFORMATION ***\n')

        for i, cam in enumerate(cam_list):

            # Retrieve TL device nodemap
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Print device information
            result &= print_device_info(nodemap_tldevice, i)

            # Adding a image list for each camera
            images[i] = list()

        # Initialize each camera
        # *** LATER ***
        # Each camera needs to be deinitialized once all images have been
        # acquired.
        for i, cam in enumerate(cam_list):

            # Initialize camera
            cam.Init()

            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(cam.GetNodeMap().GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (node retrieval; camera %d). Aborting... \n' % i)
                return False

            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry \'continuous\' retrieval %d). \
                Aborting... \n' % i)
                return False

            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Camera %d acquisition mode set to continuous...' % i)

            # Begin acquiring images
            cam.BeginAcquisition()

            print('Camera %d started acquiring images...' % i)

            print()

        # Retrieve, convert, and save images for each camera
        #
        for n in range(NUM_IMAGES):
            for i, cam in enumerate(cam_list):
                # Acquire image on all cameras
                try:
                    _thread.start_new_thread(acquire_image,(i,n))
                except Exception as e: print(e)

                # p1 = multiprocessing.Process(target=acquire_image, args=(0,n))
                # p2 = multiprocessing.Process(target=acquire_image, args=(1,n))
                #
                # # starting process 1
                # p1.start()
                # # starting process 2
                # p2.start()
                #
                # # wait until process 1 is finished
                # p1.join()
                # # wait until process 2 is finished
                # p2.join()

        # Deinitialize each camera
        for i, cam in enumerate(cam_list):
            # End acquisition
            cam.EndAcquisition()

            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()
            result &= save_list_to_avi(nodemap, nodemap_tldevice, images[i])

            # Deinitialize camera
            cam.DeInit()

            # Release reference to camera
            del cam

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result

def main():
    """
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    # Since this application saves images in the current folder
    # we must ensure that we have permission to write to this folder.
    # If we do not have permission, fail right away.
    try:
        test_file = open('test.txt', 'w+')
    except IOError:
        print('Unable to write to current directory. Please check permissions.')
        input('Press Enter to exit...')
        return False

    test_file.close()
    os.remove(test_file.name)

    result = True



    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    global cam_list
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print('Number of cameras detected: %d' % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Run on all cameras
    print('Running for all cameras...')

    result = run_multiple_cameras()

    print('complete... \n')

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    input('Done! Press Enter to exit...')
    return result

if __name__ == '__main__':
    main()