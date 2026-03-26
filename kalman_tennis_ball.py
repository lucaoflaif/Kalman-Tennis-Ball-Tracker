import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt

m2p = 2.50/1920 #meter/pixel

g = 9.81/m2p # gravity acc

lower = np.array([20, 100, 100])
upper = np.array([35, 255, 255])

cap = cv2.VideoCapture("ball.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps
#cap = cv2.VideoCapture("tennis_ball_tracker.mp4")

cX, cY, cY_tilde = (0,0,0)

# ================= KALMAN MATRICES ======================
# initial state vector
x = np.matrix([0,0,1000,2000]).T
# initial covariance matrix
P = np.matrix([[20,0,0,0],
               [0,20,0,0],
               [0,0,150,0],
               [0,0,0,200]])


F = np.matrix([[1,0,dt,0],
               [0,1,0,dt],
               [0,0,1,0],
               [0,0,0,1]])

G = np.matrix([[0, 0],
               [0, -0.5*dt**2],
               [0, 0],
               [0, -dt]])

u = np.matrix([[0,],
               [g,]])

H = np.matrix([[1,0,0,0],
               [0,1,0,0]])
I = np.eye(4)

Q = np.matrix([[1,0,0,0],
               [0,.01,0,0],
               [0,0,15,0],
               [0,0,0,50]])

R = np.matrix([[5,0],
               [0,5]])
# defining some arrays useful for plot
predicted_states = []
measured_states= []
covariances = []
frame_counter = 0
while(cap.isOpened()):
    # read the current frame
    ret, frame = cap.read()
    # is the frame available?
    if ret:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # create a yellow mask
        mask = cv2.inRange(hsv_frame, lower, upper)
        # binary image from mask, useful for moments and centroid computation 
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        M = cv2.moments(mask)
 
        # calculate x,y coordinate of center, in none found keep latest results
        if M["m00"]:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"]) 
            cY_tilde = -cY + mask.shape[0]
        else:
            continue

        #============== KALMAN =================
        # prediction
        x_pred = F @ x + G @ u
        predicted_states.append(x_pred)
        P_pred = F @ P @ F.T + Q

        # measurement
        z = np.array([cX, cY_tilde]).reshape(-1,1)
        measured_states.append((cX, cY_tilde))

        # update
        K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
        x = x_pred + K @ (z - H @ x_pred)
        P = (I - K @ H) @ P_pred @ (I-K@H).T + K@R@K.T
        covariances.append(P)

        x_pos, y_pos, x_dot, y_dot = [i.item() for i in x] 
        #============= END KALMAN ===============

        #=============== PARABOLA =================
        # calculate a, b, c from the current predicted state
        (a,b,c) = -0.5*g/ x_dot**2, y_dot/x_dot + g*x_pos/x_dot**2, y_pos - (y_dot/x_dot)*x_pos - g*(x_pos**2/(2*x_dot**2))
        parabola_points = []
        # calculate y from the parabola equation: y(frame) = ax^2 + bx + c. Keep only y that will fit the frame
        for x_frame in range(mask.shape[1]): 
            y = int(a*x_frame**2 + b*x_frame + c)
            if 0 <= y <= mask.shape[0]: parabola_points.append((x_frame, y))

        # opencv's y starts from top left, transformate the coordinate to plot it
        parabola_points = [(x_frame, -y_coord+mask.shape[0]) for (x_frame, y_coord) in parabola_points]
        
        #draws the parabola as consecutive multiple small lines
        for i in range(len(parabola_points)-1): 
            try:
                cv2.line(frame, parabola_points[i], parabola_points[i+1], (255,0,0), 5)
            except Exception as e:
                print(e)
        #=============== END PARABOLA ==============

        # draw center of tennis ball
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # write x, y position of the ball
        cv2.putText(frame, f"x:{cX}, y:{cY_tilde} [px]", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # write the full state of the ball
        cv2.putText(frame, f"x:{x_pos:.0f}, y:{y_pos:.0f} [px]; x_dot:{x_dot:.0f}, y_dot:{y_dot:.0f} [px/s]", 
                          (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # write dt
        cv2.putText(frame, f"dt: {dt:.3f} [s]", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # write frame number
        cv2.putText(frame, f"frame: {frame_counter}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # open named window, resize, show frame, wait
        cv2.namedWindow("Tennis ball tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tennis ball tracking", 1280, 720)
        cv2.imshow("Tennis ball tracking", frame)
        cv2.waitKey(int(dt*1000))

        # keep track of current frame index
        frame_counter += 1
    else:
        break

# close opencv stuff
cap.release()
cv2.destroyAllWindows()

# ================ VELOCITY COMPUTATION ==================
hand_calc_vx = []
hand_calc_vy = []

frame_list =  list(range(frame_counter-1))
for idx in range(len(frame_list)):
    hand_calc_vx.append((measured_states[idx+1][0] - measured_states[idx][0]) / dt)
    hand_calc_vy.append((measured_states[idx+1][1] - measured_states[idx][1]) / dt)
# ================ END VELOCITY COMPUTATION ==============


# =========== PLOTS ====================
plt.scatter(frame_list,hand_calc_vy,label="Computed y velocity")
plt.plot(frame_list, [x[3].item() for x in predicted_states[:-1]], label="Estimated y velocity")
plt.title("y velocity, computed visually vs estimated")
plt.xlabel("Frame number")
plt.ylabel(r"Velocity $v_y$ $[\frac{m}{s}]$")
plt.legend()
plt.show()

plt.scatter(frame_list,hand_calc_vx, label="Computed x velocity")
plt.scatter(frame_list, [x[2].item() for x in predicted_states[:-1]],label="Estimated x velocity")
plt.title("x velocity, computed visually vs estimated")
plt.xlabel("Frame number")
plt.ylabel(r"Velocity $v_x$ $[\frac{m}{s}]$")
plt.legend()
plt.show()

plt.plot(frame_list, [cov[2,2] for cov in covariances[:-1]],label=r"var($v_x$)")
plt.plot(frame_list, [cov[3,3] for cov in covariances[:-1]],label=r"var($v_y$)")
plt.title("Velocities' variances evolution")
plt.xlabel("Frame number")
plt.ylabel("Variance")
plt.legend()
plt.show()
# =========== END PLOTS ================