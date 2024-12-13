from routing_environment import RoutingEnvironmentTSN
from routing_rainbow import RoutingRainbowAgent
from config import BACKGROUND_STREAMS, VNF_LENGTH, VNF_PERIOD, VNF_DELAY, TRAINING_STEPS, \
    ROUTE_EVALUATION_EPISODES, CUSTOM_EVALUATION_EPISODES, MONITOR_TRAINING, ALL_ROUTES, CUSTOM_ROUTE
from vnf_generator import create_vnf_list, create_all_routes_vnf_lists

main_id = 'routing_cent_1'

while True:
    try:
        option = int(input('[*] Select the option you want (0 = create VNF list, 1 = train, 2 = create evaluation VNF '
                           'list, 3 = create evaluation VNF lists for all routes, 4 = evaluate routes, '
                           '5 = evaluate custom): '))
        if 0 <= option <= 4:
            break
        else:
            print('[!] Option not valid! Try again...')
    except ValueError:
        print('[!] Expected to introduce a number! Try again...')

if option == 0:
    create_vnf_list(False, ALL_ROUTES, CUSTOM_ROUTE[0], CUSTOM_ROUTE[1])
elif option == 1:
    ENV = RoutingEnvironmentTSN(main_id)

    agent = RoutingRainbowAgent(
        env=ENV,
        log_file_id=main_id
    )

    max_steps = TRAINING_STEPS
    agent.logger.info('[I] Chose training model')
    agent.logger.info('[I] Settings: time_steps = ' + str(max_steps) +
                      ' | background flows = ' + str(BACKGROUND_STREAMS) +
                      ' | replay_buffer = ' + str(agent.memory.max_size) +
                      ' | batch size = ' + str(agent.batch_size) +
                      ' | target update = ' + str(agent.target_update) + ' | gamma = ' + str(agent.gamma) +
                      ' | learning rate = ' + str(agent.learning_rate) + ' | tau = ' + str(agent.tau) +
                      ' | vnf_len = ' + str(VNF_LENGTH) + ' | vnf_prd = ' + str(VNF_PERIOD) +
                      ' | vnf_delay = ' + str(VNF_DELAY))
    agent.logger.info('[I] Extra info: ' + str(input('[*] Introduce any other setting data (just to print it): ')))
    agent.train(max_steps=max_steps, monitor_training=MONITOR_TRAINING, plotting_interval=max_steps)
elif option == 2:
    create_vnf_list(True, ALL_ROUTES, CUSTOM_ROUTE[0], CUSTOM_ROUTE[1])
elif option == 3:
    create_all_routes_vnf_lists(True)
elif option == 4:
    ENV = RoutingEnvironmentTSN(main_id)

    agent = RoutingRainbowAgent(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model routes')
    routing_model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(routing_model_name)
    agent.evaluate_routes(ROUTE_EVALUATION_EPISODES)
elif option == 5:
    ENV = RoutingEnvironmentTSN(main_id)

    agent = RoutingRainbowAgent(
        env=ENV,
        log_file_id=main_id
    )

    agent.logger.info('[I] Chose evaluating custom model')
    model_name = input('[*] Introduce the name of the model: ')
    agent.load_custom_model(model_name)
    agent.evaluate(CUSTOM_EVALUATION_EPISODES)
