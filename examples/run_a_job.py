from __future__ import absolute_import, division, print_function, unicode_literals

from autorecsys.api import Job

if __name__ == "__main__":
    config_filename = "./examples/configs/job_config.yaml"
    my_job = Job(config_filename)
    my_job.run()
