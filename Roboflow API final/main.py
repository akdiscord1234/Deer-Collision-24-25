import cv2
import roboflow
import turtle

# Initialize Roboflow model
rf = roboflow.Roboflow(api_key="NqEslJh4SM3YnifqJLzG")
workspace = rf.workspace("deerguard")
project = workspace.project("deerguard")
version = project.version(4)
model = version.model

# Setup Turtle
screen = turtle.Screen()
t = turtle.Turtle()
t.speed(0)
t.hideturtle()

def draw_classification(object_name):
    t.clear()
    t.penup()
    t.goto(-50, 0)
    t.pendown()
    t.write(f"Detected: {object_name}", font=("Arial", 16, "bold"))

# OpenCV video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame as an image to send for classification
    frame_path = "current_frame.jpg"
    cv2.imwrite(frame_path, frame)

    # Run inference
    predictions = model.predict(frame_path, hosted=False).json()

    # Check if anything was classified
    if "predictions" in predictions and len(predictions["predictions"]) > 0:
        detected_object = predictions["predictions"][0]["class"]  # Get the first detected object
        print(f"Object detected: {detected_object}")
        draw_classification(detected_object)  # Pass object name to Turtle
    else:
        print("No object detected.")
        draw_classification("None")  # Display "None" if nothing is detected

    # Show the frame
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
turtle.done()
