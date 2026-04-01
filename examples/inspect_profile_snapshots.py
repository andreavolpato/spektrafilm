"""Example: launch the Qt snapshot viewer for a processed stock."""

from spektrafilm_profile_creator.diagnostics.snapshot_viewer import launch_process_snapshot_viewer


def main():
    launch_process_snapshot_viewer('kodak_ektachrome_100')

if __name__ == '__main__':
    main()