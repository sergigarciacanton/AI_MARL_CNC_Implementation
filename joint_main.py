from joint_environment import JointEnvironmentTSN
from joint_rainbow import JointRainbowAgent
from config import CUSTOM_EVALUATION_EPISODES, ALL_ROUTES, CUSTOM_ROUTE
from vnf_generator import create_vnf_list, create_all_routes_vnf_lists

main_id = 'cent_1'

while True:
    try:
        option = int(input('[*] Select the option you want (0 = create VNF list, 1 = create evaluation VNF '
                           'list, 2 = create evaluation VNF lists for all routes, 3 = evaluate custom): '))
        if 0 <= option <= 3:
            break
        else:
            print('[!] Option not valid! Try again...')
    except ValueError:
        print('[!] Expected to introduce a number! Try again...')

if option == 0:
    create_vnf_list(False, ALL_ROUTES, CUSTOM_ROUTE[0], CUSTOM_ROUTE[1])
elif option == 1:
    create_vnf_list(True, ALL_ROUTES, CUSTOM_ROUTE[0], CUSTOM_ROUTE[1])
elif option == 2:
    create_all_routes_vnf_lists(True)
elif option == 3:
    ENV = JointEnvironmentTSN(main_id)

    agent = JointRainbowAgent(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom models')
    model_name = input('[*] Introduce the name of the routing model: ')
    agent.routing_agent.load_custom_model(model_name)
    scheduling_model_name = input('[*] Introduce the name of the scheduling model: ')
    agent.scheduling_agent.load_custom_model(scheduling_model_name)
    agent.evaluate(CUSTOM_EVALUATION_EPISODES)
