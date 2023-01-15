package video

import (
	"fmt"

	"gocv.io/x/gocv"
)

type VideoInstance struct {
	FilePath string
}

func NewVideo(videoFile string) (*VideoInstance, error) {
	videoCapture, err := gocv.VideoCaptureDevice(0)
	if err != nil {
		return nil, fmt.Errorf("failed to create new video capture device")
	}

	return nil, nil
}
