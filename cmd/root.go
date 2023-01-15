package cmd

import "github.com/spf13/cobra"

func Execute() error {
	rootCmd := &cobra.Command{
		Use:           "4bit_cam",
		Short:         "4bit_cam a process which queries a bunch of connected video devices to monitor an area",
		SilenceErrors: true,
	}
	rootCmd.AddCommand(NewClientCommand())
	return rootCmd.Execute()
}
