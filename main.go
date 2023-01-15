package main

import (
	"fmt"

	"4bit.apt_cam/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		fmt.Printf("[main] Command failed: %v\n", err)
	}
}
