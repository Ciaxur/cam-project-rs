package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var (
	cooldown *int32
)

func handleClientCmd(cmd *cobra.Command, args []string) error {
	fmt.Println("Cooldown: ", *cooldown)

	return nil
}

func NewClientCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "client",
		Short: "Starts video capture loop",
		RunE:  handleClientCmd,
	}

	cooldown = cmd.PersistentFlags().Int32("cooldown", 5, "Cooldown in minutes between notifying the user of detected image")
	cmd.MarkFlagRequired("cooldown")
	return cmd
}
