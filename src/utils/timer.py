from collections import OrderedDict, defaultdict
import rospy


class Timer(object):
    """
    Timer.
    """

    def __init__(self, time_target=0, left_shift=0, enabled=True):
        """
        Initialize Timer.
        :param time_target: target time in seconds
        :type time_target: float
        """
        self.enabled = enabled
        self._start = rospy.Time.now()
        self.leftshift(left_shift)
        self._time_target = time_target
        self.log = OrderedDict()
        self.counter = defaultdict(int)

    def restart(self):
        """
        Restart timer.
        """
        self._start = rospy.Time.now()

    def elapsed(self):
        """
        Get elapsed time since start as secs.
        :return: elapsed time.
        :rtype: float
        """
        return (rospy.Time.now() - self._start).to_sec()

    def finished(self):
        """
        Returns whether or not desired time duration has passed.
        :return: whether or not desired time duration has passed.
        :rtype: bool
        """
        return self.elapsed() >= self._time_target

    def leftshift(self, duration):
        """
        Leftshift timer.
        :param duration: duration to leftshift
        :type duration: float
        """
        self._start -= rospy.Time.from_seconds(duration)

    def log_and_restart(self, text):
        if not self.enabled: return

        if text not in self.log:
            self.log[text] = 0
        self.log[text] += self.elapsed()
        self.counter[text] += 1
        self.restart()

    def reset_log(self):
        self.log = OrderedDict()
        self.counter = OrderedDict()

    def print_log(self):
        if not self.enabled: return

        rospy.loginfo('')
        rospy.loginfo('Profiling')
        rospy.loginfo('-' * 30)
        rospy.loginfo('{:<20} {:<10}'.format('Item', 'Time (ms)'))
        rospy.loginfo('-' * 30)
        total = 0.
        for k, v in self.log.items():
            rospy.loginfo('{:<20} {:<10.4f}'.format(k, v * 1000 / self.counter[k]))
            total += v * 1000 / self.counter[k]
        rospy.loginfo('-' * 30)
        rospy.loginfo('{:<20} {:<10.4f}'.format('total', total))
        rospy.loginfo('-' * 30)
