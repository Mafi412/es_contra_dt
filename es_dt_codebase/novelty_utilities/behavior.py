import torch


class AbstractBehaviorCharacterization:
    def __init__(self, **kwargs):
        self.contained_characteristics = list()
        for key, value in kwargs.items():
            self.contained_characteristics.append(key)
            setattr(self, key, value)
    
    def compare_to(self, other_characteristic):
        assert type(self) is type(other_characteristic)
        
        if type(self) is AbstractBehaviorCharacterization:
            raise NotImplementedError()


class CombinationBehaviorCharacteristic(AbstractBehaviorCharacterization):
    def __init__(self, weights, **kwargs):
        assert weights.keys() == kwargs.keys()
        
        super().__init__(**kwargs)
        
        self.weights = weights
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        assert self.contained_characteristics == other_characteristic.contained_characteristics
        assert self.weights == other_characteristic.weights
        
        result = 0.
        for characteristic in self.contained_characteristics:
            result += self.weights[characteristic] * getattr(self, characteristic).compare_to(getattr(other_characteristic, characteristic))
            
        return result


class UniformCombinationBehaviorCharacteristic(CombinationBehaviorCharacteristic):
    def __init__(self, **kwargs):
        super().__init__({characteristic: 1/len(self.contained_characteristics) for characteristic in kwargs.keys()}, **kwargs)


class EucleidianActionHistoryCharacteristic(AbstractBehaviorCharacterization):
    def __init__(self, action_history):
        super().__init__(action_history=action_history)
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        
        history1 = self.action_history
        history2 = other_characteristic.action_history
        
        # Extend shorter history
        if history1.size(0) > history2.size(0):
            history2 = torch.cat([history2, torch.zeros((history1.size(0) - history2.size(0),) + history2.size()[1:])], dim=0)
            
        elif history1.size(0) < history2.size(0):
            history1 = torch.cat([history1, torch.zeros((history2.size(0) - history1.size(0),) + history1.size()[1:])], dim=0)
        
        return torch.linalg.vector_norm(history1 - history2).item()


class EpsilonHammingActionHistoryCharacteristic(AbstractBehaviorCharacterization):
    epsilon = 0.
    def __init__(self, action_history):
        super().__init__(action_history=action_history)
        
    def set_epsilon(new_epsilon):
        EpsilonHammingActionHistoryCharacteristic.epsilon = new_epsilon
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        
        history1 = self.action_history
        history2 = other_characteristic.action_history
        
        length_difference = abs(history1.size(0) - history2.size(0))
        
        # Reduce longer history
        if history1.size(0) > history2.size(0):
            history1 = history1[:history2.size(0)]
            
        elif history1.size(0) < history2.size(0):
            history2 = history2[:history1.size(0)]
        
        epsilon_difference = torch.logical_or(history1 > history2 + self.epsilon, history1 < history2 - self.epsilon)
        while len(epsilon_difference.size()) > 1:
            epsilon_difference = torch.any(epsilon_difference, dim=-1)
        
        return torch.sum(epsilon_difference).item() + length_difference
    
    
class NormalizedEpsilonHammingActionHistoryCharacteristic(EpsilonHammingActionHistoryCharacteristic):
    def __init__(self, action_history):
        super().__init__(action_history)
        
    def compare_to(self, other_characteristic):        
        epsilon_hamming_distance = super().compare_to(other_characteristic)
        
        max_length = max(self.action_history.size(0), other_characteristic.action_history.size(0))
        
        # We return number <= 1 expressing a portion of the action history the two behaviors were distinct.
        return epsilon_hamming_distance / max_length


class FitnessRatioCharacteristic(AbstractBehaviorCharacterization):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        
        ratio_of_fitnesses = (self.fitness / other_characteristic.fitness) \
            if self.fitness < other_characteristic.fitness \
            else (other_characteristic.fitness / self.fitness)
            
        # Ratio of fitnesses is always smaller to greater, hence it is from interval [0,1].
        return 1 - ratio_of_fitnesses


class NormalizedEndStateCharacteristic(AbstractBehaviorCharacterization):
    def __init__(self, end_state):
        super().__init__(end_state=end_state)
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        
        # We normalize the states, so now they are on the unit sphere, whence we get that their distance is <= 2 * radius = 2.
        # (We care more for the angle they hold (as vectors), which is kind of proportional to what we get, than for the absolute values of the states.)
        normalized_end_state1 = torch.nn.functional.normalize(self.end_state, dim=None)
        normalized_end_state2 = torch.nn.functional.normalize(other_characteristic.end_state, dim=None)
        
        return torch.linalg.vector_norm(normalized_end_state1 - normalized_end_state2).item() / 2


class NormalizedAverageStateCharacteristic(AbstractBehaviorCharacterization):
    def __init__(self, average_state):
        super().__init__(average_state=average_state)
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        
        # We normalize the states, so now they are on the unit sphere, whence we get that their distance is <= 2 * radius = 2.
        # (We care more for the angle they hold (as vectors), which is kind of proportional to what we get, than for the absolute values of the states.)
        normalized_average_state1 = torch.nn.functional.normalize(self.average_state, dim=None)
        normalized_average_state2 = torch.nn.functional.normalize(other_characteristic.average_state, dim=None)
        
        return torch.linalg.vector_norm(normalized_average_state1 - normalized_average_state2).item() / 2


class EndXYPositionCharacteristic(AbstractBehaviorCharacterization):
    def __init__(self, x, y):
        super().__init__(x=x, y=y)
        
    def compare_to(self, other_characteristic):
        super().compare_to(other_characteristic)
        
        return torch.linalg.vector_norm(torch.Tensor((self.x - other_characteristic.x, self.y - other_characteristic.y))).item()
