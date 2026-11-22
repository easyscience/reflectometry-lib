"""The materials palette holds the materials a user picks, never the mixtures derived from them.

A mixture's SLD is computed from its constituents, so it is not a material that can be edited
or picked for a layer. A palette entry is expected to have Parameter SLDs; a mixture in the
palette broke every consumer that reads them (a GUI's materials table among them).
"""

import json

import pytest
from easyscience import global_object

from easyreflectometry.contrasts import ReplaceFormula
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import SurfactantLayer


@pytest.fixture(autouse=True)
def clear_registry():
    global_object.registry.clear()
    yield
    global_object.registry.clear()


def _monolayer_project() -> tuple[Project, Material, Material]:
    """Air | surfactant | D2O, set through the models setter."""
    project = Project()
    air = Material(sld=0, isld=0, name='Air')
    d2o = Material(sld=6.36, isld=0, name='D2O')
    surfactant = SurfactantLayer()
    surfactant.tail_layer.solvent = air
    surfactant.head_layer.solvent = d2o
    sample = Sample(
        Multilayer(Layer(air, 0, 0, 'Air'), name='Superphase'),
        surfactant,
        Multilayer(Layer(d2o, 0, 3, 'D2O'), name='Subphase'),
    )
    project.models = ModelCollection([Model(sample=sample)])
    return project, air, d2o


def _mixtures(project: Project) -> list:
    return [material for material in project._materials if isinstance(material, MaterialMixture)]


def test_a_surfactant_model_adds_its_solvents_not_its_solvated_materials():
    project, air, d2o = _monolayer_project()

    assert _mixtures(project) == []
    assert [material.name for material in project._materials] == ['Air', 'D2O']
    assert project._materials[0] is air and project._materials[1] is d2o


def test_a_contrast_of_a_surfactant_model_adds_only_its_new_solvent():
    project, _, _ = _monolayer_project()
    acmw = Material(sld=0, isld=0, name='ACMW')
    head = project.models[0].sample[1].head_layer
    project.add_material(acmw)

    project.add_contrast(0, 'ACMW', [ReplaceFormula(head, 'C10D18NO8P')])
    project.models[1].sample[1].head_layer.solvent = acmw

    assert _mixtures(project) == []
    assert [material.name for material in project._materials] == ['Air', 'D2O', 'ACMW']


def test_a_mixture_layer_contributes_its_constituents():
    project = Project()
    si = Material(sld=2.07, isld=0, name='Si')
    a = Material(sld=1.0, isld=0, name='A')
    b = Material(sld=3.0, isld=0, name='B')
    mixture = MaterialMixture(material_a=a, material_b=b, fraction=0.3)
    sample = Sample(Multilayer(Layer(mixture, 20, 3, 'Mixed')), Multilayer(Layer(si, 0, 3, 'Si')))

    project.models = ModelCollection([Model(sample=sample)])

    assert [material.name for material in project._materials] == ['A', 'B', 'Si']


def test_a_solvent_used_only_by_a_surfactant_cannot_be_removed():
    project, _, _ = _monolayer_project()
    project.models[0].sample[2].layers[0].material = Material(sld=6.36, isld=0, name='Other D2O')

    with pytest.raises(ValueError, match='used in models'):
        project.remove_material(1)  # D2O, now only the head layer's solvent


def test_a_file_whose_palette_holds_mixtures_loads_without_them():
    project, _, _ = _monolayer_project()
    # What projects saved by earlier versions hold: the solvated materials in the palette.
    for layer in project.models[0].sample[1].layers:
        project._materials.append(layer.material)
    data = json.loads(json.dumps(project.as_dict()))
    assert len(data['materials']) == 4
    global_object.registry.clear()

    loaded = Project()
    loaded.from_dict(data)

    assert _mixtures(loaded) == []
    assert [material.name for material in loaded._materials] == ['Air', 'D2O']
    surfactant = loaded.models[0].sample[1]
    assert surfactant.head_layer.solvent is loaded._materials[1]
