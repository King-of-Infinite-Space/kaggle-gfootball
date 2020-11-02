import argparse

parser = argparse.ArgumentParser()
parser.add_argument('agents', type=str, nargs='+', help='name of agents')
parser.add_argument('-s','--scenario', type=str, default='11_vs_11_kaggle', help='name of agents')
args = parser.parse_args()

# Set up the Environment.
from kaggle_environments import make
env = make("football", debug=True, configuration={"save_video": True, "scenario_name": args.scenario, "render": True})

# output = env.run(["./miller.py", "./shev.py"])[-1]

# print('Left player: reward = %s, status = %s, info = %s' % (output[0]["reward"], output[0]["status"], output[0]["info"]))
# print('Right player: reward = %s, status = %s, info = %s' % (output[1]["reward"], output[1]["status"], output[1]["info"]))
# env.render(mode="human", width=800, height=600)
builtin_agents = ['builtin_ai', 'do_nothing', 'run_left', 'run_right']
agents = [agent if agent in builtin_agents else "./{}.py".format(agent) for agent in args.agents]
agents += ['builtin_ai']*(2-len(agents))
print(agents)
env.run(agents)