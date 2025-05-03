# create a virtual environment

python -m venv venv


## Activate the virtual environment:

On Linux/macOS (Bash/Zsh):
source venv/bin/activate


On Windows (Command Prompt):
venv\Scripts\activate

## Sample Code

curl --location 'http://localhost:5000/predict' \
--header 'Content-Type: application/json' \
--data '{"data": [[1, 85, 66, 29, 0, 26.6, 0.351, 31]]}'

