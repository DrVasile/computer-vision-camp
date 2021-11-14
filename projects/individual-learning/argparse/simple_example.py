import argparse

argparse_instance = argparse.ArgumentParser()

argparse_instance.add_argument(
    "-n",
    "--name",
    required=True,
    help="Name of the user."
)

args = vars(argparse_instance.parse_args())

print("Hello, {}!".format(args["name"]))