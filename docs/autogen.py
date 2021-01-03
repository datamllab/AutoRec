import os
import pathlib
import shutil

import keras_autodoc
import tutobooks


PAGES = {
    'preprocessor.md': [
        'autorecsys.pipeline.preprocessor.BasePreprocessor',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.format_dataset',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.load_dataset',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.transform_categorical',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.transform_numerical',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_hash_size',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_x',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_x_numerical',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_x_categorical',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_y',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_numerical_count',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.get_categorical_count',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.split_data',
        'autorecsys.pipeline.preprocessor.BasePreprocessor.preprocess',
        'autorecsys.pipeline.preprocessor.AvazuPreprocessor',
        'autorecsys.pipeline.preprocessor.AvazuPreprocessor.preprocess',
        'autorecsys.pipeline.preprocessor.CriteoPreprocessor',
        'autorecsys.pipeline.preprocessor.CriteoPreprocessor.preprocess',
        'autorecsys.pipeline.preprocessor.NetflixPrizePreprocessor',
        'autorecsys.pipeline.preprocessor.NetflixPrizePreprocessor.format_dataset',
        'autorecsys.pipeline.preprocessor.NetflixPrizePreprocessor.preprocess',
        'autorecsys.pipeline.preprocessor.MovielensPreprocessor',
        'autorecsys.pipeline.preprocessor.MovielensPreprocessor.preprocess',
    ],
    'node.md': [
        'autorecsys.pipeline.node.Input',
        'autorecsys.pipeline.node.Input.fit_transform',
        'autorecsys.pipeline.node.Input.transform',
        'autorecsys.pipeline.node.StructuredDataInput',
        'autorecsys.pipeline.node.StructuredDataInput.get_state',
        'autorecsys.pipeline.node.StructuredDataInput.set_state',
        'autorecsys.pipeline.node.StructuredDataInput.update',
        'autorecsys.pipeline.node.StructuredDataInput.infer_column_types',
    ],
    'mapper.md': [
        'autorecsys.pipeline.mapper.LatentFactorMapper',
        'autorecsys.pipeline.mapper.LatentFactorMapper.get_state',
        'autorecsys.pipeline.mapper.LatentFactorMapper.set_state',
        'autorecsys.pipeline.mapper.LatentFactorMapper.build',
        'autorecsys.pipeline.mapper.DenseFeatureMapper',
        'autorecsys.pipeline.mapper.DenseFeatureMapper.get_state',
        'autorecsys.pipeline.mapper.DenseFeatureMapper.set_state',
        'autorecsys.pipeline.mapper.DenseFeatureMapper.build',
        'autorecsys.pipeline.mapper.SparseFeatureMapper',
        'autorecsys.pipeline.mapper.SparseFeatureMapper.get_state',
        'autorecsys.pipeline.mapper.SparseFeatureMapper.set_state',
        'autorecsys.pipeline.mapper.SparseFeatureMapper.build',
    ],
    'interactor.md': [
        'autorecsys.pipeline.interactor.RandomSelectInteraction',
        'autorecsys.pipeline.interactor.RandomSelectInteraction.get_state',
        'autorecsys.pipeline.interactor.RandomSelectInteraction.set_state',
        'autorecsys.pipeline.interactor.RandomSelectInteraction.build',
        'autorecsys.pipeline.interactor.ConcatenateInteraction',
        'autorecsys.pipeline.interactor.ConcatenateInteraction.get_state',
        'autorecsys.pipeline.interactor.ConcatenateInteraction.set_state',
        'autorecsys.pipeline.interactor.ConcatenateInteraction.build',
        'autorecsys.pipeline.interactor.InnerProductInteraction',
        'autorecsys.pipeline.interactor.InnerProductInteraction.get_state',
        'autorecsys.pipeline.interactor.InnerProductInteraction.set_state',
        'autorecsys.pipeline.interactor.InnerProductInteraction.build',
        'autorecsys.pipeline.interactor.ElementwiseInteraction',
        'autorecsys.pipeline.interactor.ElementwiseInteraction.get_state',
        'autorecsys.pipeline.interactor.ElementwiseInteraction.set_state',
        'autorecsys.pipeline.interactor.ElementwiseInteraction.build',
        'autorecsys.pipeline.interactor.MLPInteraction',
        'autorecsys.pipeline.interactor.MLPInteraction.get_state',
        'autorecsys.pipeline.interactor.MLPInteraction.set_state',
        'autorecsys.pipeline.interactor.MLPInteraction.build',
        'autorecsys.pipeline.interactor.HyperInteraction',
        'autorecsys.pipeline.interactor.HyperInteraction.get_state',
        'autorecsys.pipeline.interactor.HyperInteraction.set_state',
        'autorecsys.pipeline.interactor.HyperInteraction.build',
        'autorecsys.pipeline.interactor.FMInteraction',
        'autorecsys.pipeline.interactor.FMInteraction.get_state',
        'autorecsys.pipeline.interactor.FMInteraction.set_state',
        'autorecsys.pipeline.interactor.FMInteraction.build',
        'autorecsys.pipeline.interactor.CrossNetInteraction',
        'autorecsys.pipeline.interactor.CrossNetInteraction.get_state',
        'autorecsys.pipeline.interactor.CrossNetInteraction.set_state',
        'autorecsys.pipeline.interactor.CrossNetInteraction.build',
        'autorecsys.pipeline.interactor.SelfAttentionInteraction',
        'autorecsys.pipeline.interactor.SelfAttentionInteraction.get_state',
        'autorecsys.pipeline.interactor.SelfAttentionInteraction.set_state',
        'autorecsys.pipeline.interactor.SelfAttentionInteraction.build',
    ],
    'optimizer.md': [
        'autorecsys.pipeline.optimizer.RatingPredictionOptimizer',
        'autorecsys.pipeline.optimizer.RatingPredictionOptimizer.build',
        'autorecsys.pipeline.optimizer.PointWiseOptimizer',
        'autorecsys.pipeline.optimizer.PointWiseOptimizer.build',
    ],
    'recommender.md': [
        'autorecsys.recommender.RPRecommender',
        'autorecsys.recommender.CTRRecommender',
    ],
    'auto_search.md': [
        'autorecsys.auto_search.Search',
        'autorecsys.auto_search.Search.search',
        'autorecsys.auto_search.Search.predict',
        'autorecsys.auto_search.Search.evaluate',
    ],

}


aliases_needed = [
    'tensorflow.keras.callbacks.Callback',
    'tensorflow.keras.losses.Loss',
    'tensorflow.keras.metrics.Metric',
    'tensorflow.data.Dataset'
]


ROOT = 'http://autorecsys.com/'

project_dir = pathlib.Path(__file__).resolve().parents[1]

def py_to_nb_md(dest_dir):
    for file_path in os.listdir('py/'):
        dir_path = 'py'
        file_name = file_path
        py_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != '.py':
            continue

        nb_path = os.path.join('ipynb',  file_name_no_ext + '.ipynb')
        md_path = os.path.join(dest_dir, 'tutorial', file_name_no_ext + '.md')

        tutobooks.py_to_md(py_path, nb_path, md_path, 'templates/img')

        github_repo_dir = 'keras-team/autokeras/blob/master/docs/'
        with open(md_path, 'r') as md_file:
            button_lines = [
                ':material-link: '
                "[**View in Colab**](https://colab.research.google.com/github/"
                + github_repo_dir
                + "ipynb/"
                + file_name_no_ext + ".ipynb"
                + ")   &nbsp; &nbsp;"
                # + '<span class="k-dot">•</span>'
                + ':octicons-octoface: '
                "[**GitHub source**](https://github.com/" + github_repo_dir + "py/"
                + file_name_no_ext + ".py)",
                "\n",
            ]
            md_content = ''.join(button_lines) + '\n' + md_file.read()

        with open(md_path, 'w') as md_file:
            md_file.write(md_content)


def generate(dest_dir):
    template_dir = project_dir / 'docs' / 'templates'
    doc_generator = keras_autodoc.DocumentationGenerator(
        PAGES,
        'https://github.com/datamllab/AutoRecSys',
        template_dir,
        project_dir / 'examples'
    )
    doc_generator.generate(dest_dir)
    readme = (project_dir / 'README.md').read_text()
    index = (template_dir / 'index.md').read_text()
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    (dest_dir / 'index.md').write_text(index, encoding='utf-8')
    # shutil.copyfile(project_dir / '.github' / 'CONTRIBUTING.md',
    #                 dest_dir / 'contributing.md')

    # py_to_nb_md(dest_dir)


if __name__ == '__main__':
    generate(project_dir / 'docs' / 'sources')
