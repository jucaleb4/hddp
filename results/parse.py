import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def get_time(fname):
    """ Computes total time """
    fp = open(fname, "r+")

    total_time = 0

    for line in fp.readlines():
        line = line.rstrip()

        if line.find("s]") >= 0:
            t = line[line.find("[")+1:line.find("s]")]
            t = float(t)
            total_time += t

    print("total time: {:<.0f}s ({:<.2f}h)".format(total_time, total_time/3600))
    fp.close()

def get_subarray(s, ltrig, rtrig):
    """ Returns elements within l/rtrig and returns np.array of floats """
    i = s.find(ltrig)
    j = s.find(rtrig)
    if i==-1 or j==-1: 
        return None
    xs= s[i+3:j]
    a = list(filter(lambda x : len(x) !=0, re.split("[ ]+", xs)))
    a = np.array(a, dtype=float)

    if len(a)==0:
        return None
    return a

def get_xs(fname):
    """ Gets list of x_0 from the algorithm log """
    fp = open(fname, "r+")
    n = -1
    xs = np.array([], dtype=float)

    for line in fp.readlines():
        line = line.rstrip()

        x = get_subarray(line, "x=[", "]->")

        if x is not None:
            n = len(x)
            xs = np.append(xs, x)

    xs = np.reshape(xs, newshape=(-1,n))
    fp.close()

    return xs

def animate(fname_in, fname_out, fps):
    xs = get_xs(fname_in)
    print("Starting: ", xs[0], "\n")
    print("Processing video")

    mins = np.amin(xs, axis=0)
    maxs = np.amax(xs, axis=0)

    # we will plot (0,1) and (2,3) jointly
    min_x, min_y = min(mins[0], mins[2]), min(mins[1], mins[3])
    max_x, max_y = max(maxs[0], maxs[2]), max(maxs[1], maxs[3])

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(min_x,max_x), ylim=(min_y,max_y))
    line_01, = ax.plot([], [], ls='', marker='o', color='black')
    line_23, = ax.plot([], [], ls='', marker='*', color='black')
    line_start_01, = ax.plot([xs[0,0]], [xs[0,1]], ls='', marker='o', color='red')
    line_start_23, = ax.plot([xs[0,2]], [xs[0,3]], ls='', marker='*', color='red')
    
    # initialization function: plot the background of each frame
    def init():
        line_01.set_data([], [])
        line_23.set_data([], [])
        return line_01, line_23, line_start_01, line_start_23

    # animation function.  This is called sequentially
    def animate(i):
        x1,y1 = xs[i,0], xs[i,1]
        x2,y2 = xs[i,2], xs[i,3]
        line_01.set_data(x1, y1)
        line_23.set_data(x2, y2) 
        if (i+1) % 1000 == 0:
            print("[{}]".format(i+1))
        return line_01, line_23, line_start_01, line_start_23
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=2000, interval=1000, blit=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save(fname_out, fps=24, extra_args=['-vcodec', 'libx264'])

plt.show()

def main():
    fname = "run_1_22_22_eps_5000.txt"
    # get_time(fname)
    # get_xs(fname)
    animate(fname, "video.mp4", 30)

if __name__ == "__main__":
    main()
