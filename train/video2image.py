from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import imageio
import moviepy.editor
import os
import argparse
from tqdm import tqdm
import sys


class Video2Images:

    def __init__(self,
                 video_filepath,
                 out_dir=None,
                 start_time=None,
                 end_time=None,
                 capture_rate=None,
                 save_format=".jpg",
                 ):

        """
        Args:
            video_filepath ([str]): source path of the video.
            start_time ([int], optional): Start time point of the video from\
                                           where the frames needs to be taken.
                                          Default is 0s will be taken.
            end_time ([int], optional): End time point of video from where the\
                                         frames capturing needs stop.
                                        Default is original video duration\
                                         will be taken.
            capture_rate ([int], optional): It is no. of frames which\
                                            needs to be captured per second.
                                            Default is original frame rate.
            out_dir (str, optional): The out directory where we want to\
                                     save the files. Defaults to
                                     Current working directory folder
        """

        self.video_filepath = video_filepath
        self.capture_rate = capture_rate
        self.start_time = start_time
        self.end_time = end_time
        self.save_format = save_format
        self.out_dir = out_dir

        # Capture rate should not be negative condition check
        if capture_rate is not None:
            if capture_rate <= 0:
                sys.exit("\033[1;31m Capture Frame Capture Rate cannot be <= 0 \033[00m")
                
                
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # Getting base vide filename
        video_filename = self.video_filepath.split(os.path.sep)[-1]
        # Extracting extension of input vide file
        file_extension = video_filename.split(".")[-1]
        # Valid Video Extensions
        VIDEO_EXTENSIONS = ["mov", "avi", "mpg", "mpeg", "mp4", "mkv", "wmv"]
        # Valid Output Image Extension
        IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".bmp",
                            ".tiff", ".tif", ".dicom", ".dcm"]
        # Checking for valid input video file extension
        if file_extension not in VIDEO_EXTENSIONS:
            sys.exit("\033[1;31m Input a Valid Video File \033[00m")
            
            
        # Checking for valid output image file extension format
        if self.save_format not in IMAGE_EXTENSIONS:
            sys.exit("\033[1;31m Input a valid image format \033[00m")
        # Getting duration of the original inout video
        video_duration, vformat = self.__getDuration()
        print(" Video Duration is " + f"\033[1;33m{vformat[0]}hr: {vformat[1]}min: {vformat[2]}sec \033[00m")
        # Checking if the input end time is > total duration of video or not
        if self.end_time is not None and self.end_time > video_duration:

            sys.exit("\033[1;31m End time is greater than total duration of video \033[00m")

        # Video clip extraction from original video input
        if self.start_time is not None or self.end_time is not None:

            if self.start_time is None:
                self.start_time = 0

            elif self.end_time is None:
                self.end_time = video_duration

            # Output filename for clipped video
            filename = f'{video_filename.split(".")[0]}_cut.{file_extension}'

            # Extraction of subclip function call
            ffmpeg_extract_subclip(
                self.video_filepath,
                self.start_time,
                self.end_time,
                targetname=self.out_dir + os.path.sep + filename)

            # Setting Video filepath to clipped video filepath
            self.video_filepath = os.path.join(self.out_dir, filename)

        # Initiate the video2image conversion
        self.__start()

    # Get total duration of input video
    def __getDuration(self):
        """To get overall duration of video
        Returns:
            video duration &
            [hours , minutes , seconds]
        """

        # Reading the video file
        video = moviepy.editor.VideoFileClip(self.video_filepath)

        # Getting total duration in sec
        video_duration = int(video.duration)

        seconds = video_duration

        hours = seconds // 3600
        seconds %= 3600

        mins = seconds // 60
        seconds %= 60

        return video_duration, [hours, mins, seconds]

    # The Start
    def __start(self):
        """ Method which starts the frame capturing
        """

        # Reading video
        video_reader = imageio.get_reader(self.video_filepath)

        # meta_data of video
        meta_data = video_reader.get_meta_data()

        # Getting Frames per second value
        FPS = int(meta_data['fps'])

        print(" The input Video FPS is", FPS, "frames/sec")

        if self.capture_rate is None:
            print(f" Capture rate is default FPS of input video i.e {FPS} frames/sec")
        else:
            print(" Capture rate is", self.capture_rate, "frames/sec")

        duration = meta_data['duration']

        image_count = 1

        # Capture rate after how many sec need to save the image
        if self.capture_rate is not None:

            # Check if capture is >= to the input video FPS
            if self.capture_rate > FPS:

                sys.exit("\033[1;31m Capture rate cannot be greater than maximum FPS of input video \033[00m")

        else:

            # capture all frames
            self.capture_rate = FPS

        # Cycle counter
        counter = 0

        # Cycle is equal to capture rate by user
        cycle = self.capture_rate

        # Iterate over entire frames in video
        for _, img in tqdm(enumerate(video_reader),
                           desc="\033[1;37m Capturing Frames... \033[00m",
                           unit="iter",
                           total=int(round(duration)*meta_data['fps'])):

            # counter increment
            counter += 1

            # if counter is less than cycle
            if counter <= cycle:

                imageio.imsave(
                    os.path.join(self.out_dir, f'{image_count}{self.save_format}'),
                    im=img
                    )

                image_count += 1

            # if counter is equal to the last frame then reset the counter
            if counter == FPS:

                counter = 0

        print(f"\033[1;32;40m Done. Total frames captured: {image_count-1} \033[00m")


def video2image(video_root:str, output_folder:str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_class in os.listdir(video_root):
        class_path = os.path.join(video_root, video_class)
        output_class_path = os.path.join(output_folder, video_class)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            print(output_class_path)
            Video2Images(video_filepath = video_path, out_dir = os.path.normpath(output_class_path))


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", type=str, required=True, help="path to directory containing images stored by label.")
    parser.add_argument("-n", "--name", type=str, required=True, help="output dir name",default='output')
    args = parser.parse_args()

    input = args.input
    name = args.name


    video2image(video_root=input,output_folder=name)