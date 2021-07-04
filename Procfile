RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python


mkdir -p ~/.streamlit/ 
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
