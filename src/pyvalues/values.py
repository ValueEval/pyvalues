from abc import ABC, abstractmethod
from typing import Annotated, Callable, Iterable
from pydantic import BaseModel, ConfigDict, Field

Score = Annotated[float, Field(ge=0, le=1)]


class AttainmentScore(BaseModel):
    attained: Score = 0.0
    constrained: Score = 0.0

    def total(self) -> Score:
        score = self.attained + self.constrained
        assert score >= 0
        assert score <= 1
        return score


def combine_attainment_scores(
        scores: Iterable[AttainmentScore],
        mode: Callable[[Iterable[float]], float] = max
    ) -> AttainmentScore:
    totals = []
    attained = 0.0
    constrained = 0.0
    for score in scores:
        totals.append(score.total())
        attained += score.attained
        constrained += score.constrained
    total = mode(totals)
    assert total >= 0
    assert total <= 1
    if attained + constrained == 0:
        return AttainmentScore()
    else:
        weighted_attained = total * (attained / (attained + constrained))
        weighted_constrained = total - weighted_attained
        return AttainmentScore(
            attained=weighted_attained,
            constrained=weighted_constrained
        )


class Values(ABC, BaseModel):
    pass


class ValuesWithoutAttainment(Values):
    pass


class ValuesWithAttainment(Values):

    @abstractmethod
    def without_attainment(self) -> ValuesWithoutAttainment:
        pass


class OriginalValues(ValuesWithoutAttainment):
    self_direction: Score = Field(serialization_alias="Self-direction", default=0.0)
    stimulation: Score = Field(serialization_alias="Stimulation", default=0.0)
    hedonism: Score = Field(serialization_alias="Hedonism", default=0.0)
    achievement: Score = Field(serialization_alias="Achievement", default=0.0)
    power: Score = Field(serialization_alias="Power", default=0.0)
    security: Score = Field(serialization_alias="Security", default=0.0)
    tradition: Score = Field(serialization_alias="Tradition", default=0.0)
    conformity: Score = Field(serialization_alias="Conformity", default=0.0)
    benevolence: Score = Field(serialization_alias="Benevolence", default=0.0)
    universalism: Score = Field(serialization_alias="Universalism", default=0.0)

    model_config = ConfigDict(serialize_by_alias=True)


class RefinedCoarseValues(ValuesWithoutAttainment):
    self_direction: Score = Field(serialization_alias="Self-direction", default=0.0)
    stimulation: Score = Field(serialization_alias="Stimulation", default=0.0)
    hedonism: Score = Field(serialization_alias="Hedonism", default=0.0)
    achievement: Score = Field(serialization_alias="Achievement", default=0.0)
    power: Score = Field(serialization_alias="Power", default=0.0)
    face: Score = Field(serialization_alias="Face", default=0.0)
    security: Score = Field(serialization_alias="Security", default=0.0)
    tradition: Score = Field(serialization_alias="Tradition", default=0.0)
    conformity: Score = Field(serialization_alias="Conformity", default=0.0)
    humility: Score = Field(serialization_alias="Humility", default=0.0)
    benevolence: Score = Field(serialization_alias="Benevolence", default=0.0)
    universalism: Score = Field(serialization_alias="Universalism", default=0.0)

    model_config = ConfigDict(serialize_by_alias=True)

    def original_values(self) -> OriginalValues:
        return OriginalValues(
            self_direction=self.self_direction,
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=self.power,
            security=self.security,
            tradition=self.tradition,
            conformity=self.conformity,
            benevolence=self.benevolence,
            universalism=self.universalism,
        )


class RefinedValues(ValuesWithoutAttainment):
    self_direction_thought: Score = Field(serialization_alias="Self-direction: thought", default=0.0)
    self_direction_action: Score = Field(serialization_alias="Self-direction: action", default=0.0)
    stimulation: Score = Field(serialization_alias="Stimulation", default=0.0)
    hedonism: Score = Field(serialization_alias="Hedonism", default=0.0)
    achievement: Score = Field(serialization_alias="Achievement", default=0.0)
    power_dominance: Score = Field(serialization_alias="Power: dominance", default=0.0)
    power_resources: Score = Field(serialization_alias="Power: resources", default=0.0)
    face: Score = Field(serialization_alias="Face", default=0.0)
    security_personal: Score = Field(serialization_alias="Security: personal", default=0.0)
    security_societal: Score = Field(serialization_alias="Security: societal", default=0.0)
    tradition: Score = Field(serialization_alias="Tradition", default=0.0)
    conformity_rules: Score = Field(serialization_alias="Conformity: rules", default=0.0)
    conformity_interpersonal: Score = Field(serialization_alias="Conformity: interpersonal", default=0.0)
    humility: Score = Field(serialization_alias="Humility", default=0.0)
    benevolence_caring: Score = Field(serialization_alias="Benevolence: caring", default=0.0)
    benevolence_dependability: Score = Field(serialization_alias="Benevolence: dependability", default=0.0)
    universalism_concern: Score = Field(serialization_alias="Universalism: concern", default=0.0)
    universalism_nature: Score = Field(serialization_alias="Universalism: nature", default=0.0)
    universalism_tolerance: Score = Field(serialization_alias="Universalism: tolerance", default=0.0)

    model_config = ConfigDict(serialize_by_alias=True)

    def coarse_values(self, mode: Callable[[Iterable[float]], float] = max) -> RefinedCoarseValues:
        return RefinedCoarseValues(
            self_direction=mode([self.self_direction_action, self.self_direction_thought]),
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=mode([self.power_dominance, self.power_resources]),
            face=self.face,
            security=mode([self.security_personal, self.security_societal]),
            tradition=self.tradition,
            conformity=mode([self.conformity_rules, self.conformity_interpersonal]),
            humility=self.humility,
            benevolence=mode([self.benevolence_caring, self.benevolence_dependability]),
            universalism=mode([self.universalism_concern, self.universalism_nature, self.universalism_tolerance])
        )
    
    def original_values(self, mode: Callable[[Iterable[float]], float] = max) -> OriginalValues:
        return self.coarse_values(mode=mode).original_values()


class OriginalValuesWithAttainment(ValuesWithAttainment):
    self_direction: AttainmentScore = Field(serialization_alias="Self-direction", default=AttainmentScore())
    stimulation: AttainmentScore = Field(serialization_alias="Stimulation", default=AttainmentScore())
    hedonism: AttainmentScore = Field(serialization_alias="Hedonism", default=AttainmentScore())
    achievement: AttainmentScore = Field(serialization_alias="Achievement", default=AttainmentScore())
    power: AttainmentScore = Field(serialization_alias="Power", default=AttainmentScore())
    security: AttainmentScore = Field(serialization_alias="Security", default=AttainmentScore())
    tradition: AttainmentScore = Field(serialization_alias="Tradition", default=AttainmentScore())
    conformity: AttainmentScore = Field(serialization_alias="Conformity", default=AttainmentScore())
    benevolence: AttainmentScore = Field(serialization_alias="Benevolence", default=AttainmentScore())
    universalism: AttainmentScore = Field(serialization_alias="Universalism", default=AttainmentScore())

    model_config = ConfigDict(serialize_by_alias=True)

    def without_attainment(self) -> OriginalValues:
        return OriginalValues(
            self_direction=self.self_direction.total(),
            stimulation=self.stimulation.total(),
            hedonism=self.hedonism.total(),
            achievement=self.achievement.total(),
            power=self.power.total(),
            security=self.security.total(),
            tradition=self.tradition.total(),
            conformity=self.conformity.total(),
            benevolence=self.benevolence.total(),
            universalism=self.universalism.total(),
        )


class RefinedCoarseValuesWithAttainment(ValuesWithAttainment):
    self_direction: AttainmentScore = Field(serialization_alias="Self-direction", default=AttainmentScore())
    stimulation: AttainmentScore = Field(serialization_alias="Stimulation", default=AttainmentScore())
    hedonism: AttainmentScore = Field(serialization_alias="Hedonism", default=AttainmentScore())
    achievement: AttainmentScore = Field(serialization_alias="Achievement", default=AttainmentScore())
    power: AttainmentScore = Field(serialization_alias="Power", default=AttainmentScore())
    face: AttainmentScore = Field(serialization_alias="Face", default=AttainmentScore())
    security: AttainmentScore = Field(serialization_alias="Security", default=AttainmentScore())
    tradition: AttainmentScore = Field(serialization_alias="Tradition", default=AttainmentScore())
    conformity: AttainmentScore = Field(serialization_alias="Conformity", default=AttainmentScore())
    humility: AttainmentScore = Field(serialization_alias="Humility", default=AttainmentScore())
    benevolence: AttainmentScore = Field(serialization_alias="Benevolence", default=AttainmentScore())
    universalism: AttainmentScore = Field(serialization_alias="Universalism", default=AttainmentScore())

    model_config = ConfigDict(serialize_by_alias=True)

    def original_values(self) -> OriginalValuesWithAttainment:
        return OriginalValuesWithAttainment(
            self_direction=self.self_direction,
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=self.power,
            security=self.security,
            tradition=self.tradition,
            conformity=self.conformity,
            benevolence=self.benevolence,
            universalism=self.universalism,
        )

    def without_attainment(self) -> RefinedCoarseValues:
        return RefinedCoarseValues(
            self_direction=self.self_direction.total(),
            stimulation=self.stimulation.total(),
            hedonism=self.hedonism.total(),
            achievement=self.achievement.total(),
            power=self.power.total(),
            face=self.face.total(),
            security=self.security.total(),
            tradition=self.tradition.total(),
            conformity=self.conformity.total(),
            humility=self.humility.total(),
            benevolence=self.benevolence.total(),
            universalism=self.universalism.total(),
        )


class RefinedValuesWithAttainment(ValuesWithAttainment):
    self_direction_thought: AttainmentScore = Field(
        serialization_alias="Self-direction: thought", default=AttainmentScore())
    self_direction_action: AttainmentScore = Field(
        serialization_alias="Self-direction: action", default=AttainmentScore())
    stimulation: AttainmentScore = Field(
        serialization_alias="Stimulation", default=AttainmentScore())
    hedonism: AttainmentScore = Field(
        serialization_alias="Hedonism", default=AttainmentScore())
    achievement: AttainmentScore = Field(
        serialization_alias="Achievement", default=AttainmentScore())
    power_dominance: AttainmentScore = Field(
        serialization_alias="Power: dominance", default=AttainmentScore())
    power_resources: AttainmentScore = Field(
        serialization_alias="Power: resources", default=AttainmentScore())
    face: AttainmentScore = Field(
        serialization_alias="Face", default=AttainmentScore())
    security_personal: AttainmentScore = Field(
        serialization_alias="Security: personal", default=AttainmentScore())
    security_societal: AttainmentScore = Field(
        serialization_alias="Security: societal", default=AttainmentScore())
    tradition: AttainmentScore = Field(
        serialization_alias="Tradition", default=AttainmentScore())
    conformity_rules: AttainmentScore = Field(
        serialization_alias="Conformity: rules", default=AttainmentScore())
    conformity_interpersonal: AttainmentScore = Field(
        serialization_alias="Conformity: interpersonal", default=AttainmentScore())
    humility: AttainmentScore = Field(
        serialization_alias="Humility", default=AttainmentScore())
    benevolence_caring: AttainmentScore = Field(
        serialization_alias="Benevolence: caring", default=AttainmentScore())
    benevolence_dependability: AttainmentScore = Field(
        serialization_alias="Benevolence: dependability", default=AttainmentScore())
    universalism_concern: AttainmentScore = Field(
        serialization_alias="Universalism: concern", default=AttainmentScore())
    universalism_nature: AttainmentScore = Field(
        serialization_alias="Universalism: nature", default=AttainmentScore())
    universalism_tolerance: AttainmentScore = Field(
        serialization_alias="Universalism: tolerance", default=AttainmentScore())

    model_config = ConfigDict(serialize_by_alias=True)

    def coarse_values(self, mode: Callable[[Iterable[float]], float] = max) -> RefinedCoarseValuesWithAttainment:
        return RefinedCoarseValuesWithAttainment(
            self_direction=combine_attainment_scores(
                [self.self_direction_action, self.self_direction_thought], mode=mode),
            stimulation=self.stimulation,
            hedonism=self.hedonism,
            achievement=self.achievement,
            power=combine_attainment_scores(
                [self.power_dominance, self.power_resources], mode=mode),
            face=self.face,
            security=combine_attainment_scores(
                [self.security_personal, self.security_societal], mode=mode),
            tradition=self.tradition,
            conformity=combine_attainment_scores(
                [self.conformity_rules, self.conformity_interpersonal], mode=mode),
            humility=self.humility,
            benevolence=combine_attainment_scores(
                [self.benevolence_caring, self.benevolence_dependability], mode=mode),
            universalism=combine_attainment_scores(
                [self.universalism_concern, self.universalism_nature, self.universalism_tolerance], mode=mode)
        )
    
    def original_values(self, mode: Callable[[Iterable[float]], float] = max) -> OriginalValuesWithAttainment:
        return self.coarse_values(mode=mode).original_values()

    def without_attainment(self) -> RefinedValues:
        return RefinedValues(
            self_direction_action=self.self_direction_action.total(),
            self_direction_thought=self.self_direction_thought.total(),
            stimulation=self.stimulation.total(),
            hedonism=self.hedonism.total(),
            achievement=self.achievement.total(),
            power_dominance=self.power_dominance.total(),
            power_resources=self.power_resources.total(),
            face=self.face.total(),
            security_personal=self.security_personal.total(),
            security_societal=self.security_societal.total(),
            tradition=self.tradition.total(),
            conformity_rules=self.conformity_rules.total(),
            conformity_interpersonal=self.conformity_interpersonal.total(),
            humility=self.humility.total(),
            benevolence_caring=self.benevolence_caring.total(),
            benevolence_dependability=self.benevolence_dependability.total(),
            universalism_concern=self.universalism_concern.total(),
            universalism_nature=self.universalism_nature.total(),
            universalism_tolerance=self.universalism_tolerance.total(),
        )
