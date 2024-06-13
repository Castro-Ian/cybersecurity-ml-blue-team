# Use an official Nginx image
FROM nginx:alpine

# Copy the static files and templates to the appropriate Nginx directory
COPY static /usr/share/nginx/html/static
COPY templates /usr/share/nginx/html/templates
COPY nginx.conf /etc/nginx/nginx.conf
