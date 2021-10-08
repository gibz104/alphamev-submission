# pip3 install pandas numpy ast pycaret=2.3.3 tune-sklearn ray[tune] scikit-optimize

import pandas as pd
import numpy as np
import ast
import pycaret
from pycaret.classification import *
from pycaret.regression import *

contractDict = {
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": "Uniswap: Uniswap Protocol: UNI Token",
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Uniswap: Uniswap V2: Router 2",
    "0x090d4613473dee047c3f2706764f49e0821d256e": "Uniswap: Uniswap: Token Distributor",
    "0xc2edad668740f1aa35e4d8f227fb8e17dca888cd": "SushiSwap: SushiSwap: MasterChef LP Staking Pool",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "SushiSwap: SushiSwap: Router",
    "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2": "SushiSwap: SushiSwap: SUSHI Token",
    "0x088ee5007c98a9677165d78dd2109ae4a3d04d0c": "SushiSwap: SushiSwap: YFI",
    "0x3e66b66fd1d0b02fda6c811da9e0547970db2f21": "Balancer: Balancer: Exchange Proxy 2",
    "0xba100000625a3754423978a60c9317c58a424e3d": "Balancer: Balancer: BAL Token",
    "0x9008D19f58AAbD9eD0D60971565AA8510560ab41": "CowSwap: Settlement Contract",
    "0x3328f5f2cEcAF00a2443082B657CedEAf70bfAEf": "CowSwap: OLD Settlement Contract",
    "0xe41d2489571d322189246dafa5ebde1f4699f498": "ZRX: ZRX Token",
    "0xd26114cd6EE289AccF82350c8d8487fedB8A0C07": "OMG Network: OMG Token",
    "0x111111111117dc0aa78b770fa6a738034120c302": "1INCH: 1INCH Token",
    "0x3A8cCCB969a61532d1E6005e2CE12C200caeCe87": "TitanSwap: Titan Token",
    "0x6c28AeF8977c9B773996d0e8376d2EE379446F2f": "Quickswap: QUICK Token",
    "0xdd974d5c2e2928dea5f71b9825b8b646686bd200": "Kyber: Old KNC Token",
    "0x9aab3f75489902f3a48495025729a0af77d4b11e": "Kyber: Kyber Proxy 2",
    "0xecf0bdb7b3f349abfd68c3563678124c5e8aaea3": "Kyber: Kyber Staking",
    "0xdeFA4e8a7bcBA345F687a2f1456F5Edd9CE97202": "Kyber: Kyber Network Crystal v2",
    "0xbbbbca6a901c926f240b89eacb641d8aec7aeafd": "Loopring: LRC Token",
    "0x0baba1ad5be3a5c0a66e7ac838a129bf948f1ea4": "Loopring: Exchange V2",
    "0xf4662bb1c4831fd411a95b8050b3a5998d8a4a5b": "Loopring: Staking Pool",
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": "Aave: AAVE Token",
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave: Aave: Lending Pool V2",
    "0xdcd33426ba191383f1c9b431a342498fdac73488": "Aave: WETH Gateway",
    "0x030ba81f1c18d280636f32af80b9aad02cf0854e": "Aave: aWETH Token",
    "0xbcca60bb61934080951369a648fb03df4f96263c": "Aave: aUSDC Token",
    "0x028171bca77440897b824ca71d1c56cac55b68a3": "Aave: aDAI Token",
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "Compound: Compcontroller",
    "0xc00e94cb662c3520282e6f5717214004a7f26888": "Compound: COMP Token",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "Maker: Dai Stablecoin",
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": "Maker: Maker Token",
    "0x4c19596f5aaff459fa38b0f7ed92f11ae6543784": "TrueFi: TrueFi Token",
    "0x2ba592f78db6436527729929aaf6c908497cb200": "CREAM Finance: CREAM Token",
    "0x3d5bc3c8d13dcb8bf317092d84783c2697ae9258": "CREAM Finance: Comptroller",
    "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f": "Synthetix: SNX Token",
    "0xb440dd674e1243644791a4adfe3a2abb0a92d309": "Synthetix: Fee Pool",
    "0xd7c49cee7e9188cca6ad8ff264c1da2e69d4cf3b": "Nexus Mutual: NXM Token",
    "0x84edffa16bb0b9ab1163abb0a13ff0744c11272f": "Nexus Mutual: Pooled Staking",
    "0x1e0447b19bb6ecfdae1e4ae1694b0c3659614e4e": "dYdX: Solo Margin",
    "0xa8b39829ce2246f89b31c013b8cde15506fb9a76": "dYdX: Pay Proxy for Solo Margin",
    "0xd54f502e184b6b739d7d27a6410a67dc462d69c8": "dYdX: L2 Perp Smart Contract",
    "0x09403fd14510f8196f7879ef514827cd76960b5d": "dYdX: Perp Proxy",
    "0x8129b737912e17212c8693b781928f5d0303390a": "dYdX: L2 On-Chain Operator",
    "0x39246c4f3f6592c974ebc44f80ba6dc69b817c71": "Opyn: Options Exchange",
    "0xcc5d905b9c2c8c9329eb4e25dc086369d6c7777c": "Opyn: Options Factory",
    "0x6123b0049f904d730db3c36a31167d9d4121fa6b": "Ribbon Finance: RBN Token",
    "0x722122df12d4e14e13ac3b6895a86e84145b6967": "Tornado Cash: Tornado Cash Proxy",
    "0x77777feddddffc19ff86db637967013e6c6a116c": "Tornado Cash: TORN Token",
    "0x746aebc06d2ae31b71ac51429a19d54e797878e9": "Tornado Cash: Mining v2",
    "0xa160cdab225685da1d56aa342ad8841c3b53f291": "Tornado Cash: 100 ETH",
    "0x910cbd523d972eb0a6f4cae4618ad62622b39dbf": "Tornado Cash: 10 ETH",
    "0x47ce0c6ed5b0ce3d3a51fdb1c52dc66a7c3c2936": "Tornado Cash: 1 ETH",
    "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc": "Tornado Cash: 0.1 ETH",
    "0x4a57e687b9126435a9b19e4a802113e266adebde": "Flexa: FXC Token",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC: USDC Token",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT: USDT Token",
    "0x4fabb145d64652a948d72533023f6e7a623c7c53": "BUSD: Binance USD",
    "0xa47c8bf37f92abed4a126bda807a7b7498661acd": "WUSDT: Wrapped USDT",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "WETH: Wrapped ETH"
}

def generate_features(dataset):
    features = []
    callData = []
    for index, row in dataset[['txData', 'txTrace']].iterrows():
        txData = ast.literal_eval(row['txData'])
        txTrace = ast.literal_eval(row['txTrace'])
        callData.append(aggregateCallData(txTrace))

        features.append([
            int(txData['from'], 0) % (2 ** 30),
            (int(txData['to'], 0) if txData['to'] is not None else 0) % (2 ** 30),
            int(txData['gas'], 0),
            int(txData['gasPrice'], 0) / (10 ** 9),
            (int(txData['input'][:10], 0) if txData['input'] != '0x' else 0) % (2 ** 30),
            (int(len(txData['input'][10:])) / 32 if txData['input'] != '0x' else 0),
            int(txData['nonce'], 0),
            int(txData['value'], 0) / (10 ** 18),

            int(txTrace['gas'], 0),
            (int(len(txTrace['output'])) - 2 if 'output' in txTrace.keys() else 0),
            (int(txTrace['gasUsed'], 0) if 'gasUsed' in txTrace.keys() else 0)
        ])

    mainFeatures = pd.DataFrame(np.array(features), columns=['from', 'to', 'gasLimit', 'gasPrice', 'inputMethod',
                                                             'inputSize', 'nonce', 'value', 'txTraceGas', 'outputSize',
                                                             'gasUsed'])
    callFeatureNames = ['totalCalls', 'nCALL', 'nSTATICCALL', 'nDELEGATECALL', 'nCREATE',
                        'nSELFDESTRUCT', 'totalValue', 'totalInputSize', 'totalOutputSize',
                        'callsGasUsed', 'nErrors', 'errExecRev', 'errOutOfGas', 'errBadInst',
                        'errBadJumpDest'] + list(contractDict.values())
    callFeatures = pd.DataFrame(np.array(callData), columns=callFeatureNames)

    return pd.concat([mainFeatures, callFeatures], axis=1)

def aggregateCallData(trace):
    callData = {
        'totalCalls': 0,
        'nCALL': 0,
        'nSTATICCALL': 0,
        'nDELEGATECALL': 0,
        'nCREATE': 0,
        'nSELFDESTRUCT': 0,
        'totalValue': 0,
        'totalInputSize': 0,
        'totalOutputSize': 0,
        'callsGasUsed': 0,
        'nErrors': 0,
        'errExecRev': 0,
        'errOutOfGas': 0,
        'errBadInst': 0,
        'errBadJumpDest': 0
    }
    contract_count = dict.fromkeys(list(contractDict.keys()), 0)

    def recurseCalls(trace):
        if 'calls' in trace.keys():
            for intTrace in trace['calls']:
                recurseCalls(intTrace)

        callData['totalCalls'] += 1
        if trace['type'] == 'CALL':
            callData['nCALL'] += 1
        elif trace['type'] == 'STATICCALL':
            callData['nSTATICCALL'] += 1
        elif trace['type'] == 'DELEGATECALL':
            callData['nDELEGATECALL'] += 1
        elif trace['type'] == 'CREATE':
            callData['nCREATE'] += 1
        elif trace['type'] == 'SELFDESTRUCT':
            callData['nSELFDESTRUCT'] += 1

        callData['totalValue'] += (int(trace['value'], 0) / (10 ** 18) if 'value' in trace.keys() else 0)
        callData['totalInputSize'] += (int(len(trace['input'])) - 2 if 'input' in trace.keys() else 0)
        callData['totalOutputSize'] += (int(len(trace['output'])) - 2 if 'output' in trace.keys() else 0)
        callData['callsGasUsed'] += (int(trace['gasUsed'], 0) if 'gasUsed' in trace.keys() else 0)

        if 'error' in trace.keys():
            callData['nErrors'] += 1
            if trace['error'] == 'execution reverted':
                callData['errExecRev'] += 1
            elif trace['error'] == 'Out of gas':
                callData['errOutOfGas'] += 1
            elif trace['error'] == 'Bad instruction':
                callData['errBadInst'] += 1
            elif trace['error'] == 'Bad jump destination':
                callData['errBadJumpDest'] += 1

        if 'to' in trace.keys():
            if trace['to'] in contract_count:
                contract_count[trace['to']] += 1

    recurseCalls(trace)
    return list(callData.values()) + list(contract_count.values())

def predictClassification():
    #### DATA LOAD ####
    train = pd.read_csv('train.csv')
    train_features = generate_features(train)
    test = pd.read_csv('test.csv')
    test_features = generate_features(test)

    #### CLASSIFICATION ####
    data = pd.concat([train_features, train['Label0']], axis=1)
    grid = pycaret.classification.setup(data=data,
                                        target='Label0',
                                        normalize=True,
                                        normalize_method='minmax',
                                        fold_shuffle=True,
                                        remove_outliers=True,
                                        feature_selection=True,
                                        fix_imbalance=True,
                                        fold=10,
                                        html=False,
                                        silent=True)

    top5 = pycaret.classification.compare_models(n_select=5, sort='AUC')
    tuned_top5 = [pycaret.classification.tune_model(i,
                                                    optimize='AUC',
                                                    n_iter=10,
                                                    search_library='tune-sklearn',
                                                    search_algorithm='bayesian',
                                                    early_stopping=True,
                                                    choose_better=True) for i in top5]
    bagged_top5 = [pycaret.classification.ensemble_model(i, optimize='AUC') for i in tuned_top5]
    blender = pycaret.classification.blend_models(estimator_list=top5, optimize='AUC', method='soft')
    best_classification_model = pycaret.classification.automl(optimize='AUC')
    classificationPredictions = pycaret.classification.predict_model(best_classification_model, data=test_features, raw_score=True)['Score_True']
    return classificationPredictions

def predictRegression():
    #### DATA LOAD ####
    train = pd.read_csv('train.csv')
    train_features = generate_features(train)
    test = pd.read_csv('test.csv')
    test_features = generate_features(test)

    #### REGRESSION ####
    data = pd.concat([train_features, train[['Label0', 'Label1']]], axis=1)
    data = data[data['Label0'] == True]
    data = data.drop(columns=['Label0'])
    grid = pycaret.regression.setup(data=data,
                                    target='Label1',
                                    normalize=True,
                                    normalize_method='minmax',
                                    fold_shuffle=True,
                                    remove_outliers=True,
                                    feature_selection=True,
                                    feature_selection_method='boruta',
                                    fold=10,
                                    html=False,
                                    silent=True)

    top5 = pycaret.regression.compare_models(n_select=5, sort='MSE')
    tuned_top5 = [pycaret.regression.tune_model(i,
                                                optimize='MSE',
                                                n_iter=10,
                                                search_library='tune-sklearn',
                                                search_algorithm='bayesian',
                                                early_stopping=True,
                                                choose_better=True) for i in top5]
    bagged_top5 = [pycaret.regression.ensemble_model(i, optimize='MSE') for i in tuned_top5]
    blender = pycaret.regression.blend_models(estimator_list=top5, optimize='MSE')
    best_regression_model = pycaret.regression.automl(optimize='MSE')
    regressionPredictions = pycaret.regression.predict_model(best_regression_model, data=test_features)['Label']
    return regressionPredictions

#### CREATE submission.csv ####
pd.concat([predictClassification(), predictRegression()], axis=1).to_csv('submission.csv', encoding='utf-8', header=False, index=False)
