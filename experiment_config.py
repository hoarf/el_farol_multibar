from wlu import Experiment
import numpy as np
import json
import time

# 0: Print nothing
# 1+: Print everything
VERBOSITY_LEVEL = 0

na = 101
ts = [[.3,.5],[.2,.1,.1,.3]]
decay_ps = [("partial",.5), ("exponential",.9), (None,.3), (None,.5)]
alphas = [.2, .02]
init_q_values = ["rand", "zeros"]
nw = 5

def prt(str, level=0):
  if VERBOSITY_LEVEL > level:
    print(str)

class NumpyAwareJSONEncoder(json.JSONEncoder):

  def default(self, obj):
    if isinstance(obj, np.ndarray):
      if obj.ndim == 1:
        return obj.tolist()
      else:
        return [self.default(obj[i]) for i in range(obj.shape[0])]
    return json.JSONEncoder.default(self, obj)

class ElFarolDataSerializer():

  def __init__(self, filename):
    self.filename = filename

  def save_data(self):
    for t_ix, t in enumerate(ts):
      prt("t=%s" % t)
      for d_p_ix, (d, p) in enumerate(decay_ps):
        prt("decay=%s, p=%s" % (d, p))
        for a_ix, a in enumerate(alphas):
          prt("alpha=%s" % a)
          for i_ix, i in enumerate(init_q_values):
            prt("init_q_value=%s" % i)
            experiment = Experiment(nr_weeks=nw, thresholds=t, nr_agents=na,
                            debug=False, use_wlu=False, alpha=a, p=p, decay=d,
                            init_q_value=i, continuous=True)

            result = experiment.run()
            # serialized keys
            s_ix = (str(t_ix), str(d_p_ix), str(a_ix), str(i_ix))
            self.data[s_ix[0]] = { s_ix[1]: { s_ix[2]: { s_ix[3]: result }}}

  def run_and_save_data(self):

    self.data = {
      'metadata': {
        'time': time.strftime('%c'),
        'number_weeks': nw,
        'number_agents': na
      }
    }

    self.save_data()

    with open(self.filename, 'w') as f:
      f.write(json.dumps(self.data, cls=NumpyAwareJSONEncoder))

  def load_data(self):
    with open(self.filename, 'r') as f:
      return json.load(f)

if __name__ == '__main__':
  ds = ElFarolDataSerializer('data.json')
  ds.run_and_save_data()
  print(ds.load_data())
