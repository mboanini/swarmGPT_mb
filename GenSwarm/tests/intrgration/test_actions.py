# """
# Copyright (c) 2024 WindyLab of Westlake University, China
# All rights reserved.

# This software is provided "as is" without warranty of any kind, either
# express or implied, including but not limited to the warranties of
# merchantability, fitness for a particular purpose, or non-infringement.
# In no event shall the authors or copyright holders be liable for any
# claim, damages, or other liability, whether in an action of contract,
# tort, or otherwise, arising from, out of, or in connection with the
# software or the use or other dealings in the software.
# """

# import unittest
# import os
# import shutil
# import asyncio

# from modules.file import logger, File
# from modules.framework.actions import *
# from modules.framework.code import FunctionTree
# from modules.framework.constraint import ConstraintPool
# from modules.utils import root_manager


# def empty_folder(folder_path):
#     if not os.path.exists(folder_path):
#         return
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")


# def copy_files(path):
#     project_root = root_manager.project_root
#     util_file = File(root=os.path.join(project_root, "modules/env"), name="apis.py")
#     util_file.copy(path)
#     run_file = File(root=os.path.join(project_root, "modules/env"), name="run.py")
#     run_file.copy(path)


# class TestAction(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls._path = os.path.dirname(os.path.abspath(__file__)) + "/workspace"
#         root_manager.update_root(cls._path)
#         logger.set_file(File("log.md"))

#     def setUp(self):
#         self._constraint_pool = ConstraintPool()
#         self._function_pool = FunctionTree("test")
#         self._constraint_pool.reset()
#         # self._function_pool.reset()
#         print("Running test:", self._testMethodName)

#     def tearDown(self) -> None:
#         self._constraint_pool.reset()
#         # self._function_pool.reset()

#     def test_1_analyze_constraint(self):
#         empty_folder(self._path)
#         copy_files(self._path)
#         analyst = AnalyzeConstraints("constraints")
#         analyst.context.command = (
#             "Form a flock with other robots, navigating together by keeping aligned, spaced out, "
#             "and cohesive. Avoid obstacles and stay away from the environment's edges and obstacles."
#         )
#         self.assertEqual(len(self._constraint_pool._constraint_nodes), 0)

#         asyncio.run(analyst.run())
#         self.assertGreater(len(self._constraint_pool._constraint_nodes), 0)
#         analyst.context.save_to_file(f"{self._path}/analyze_constraints.pkl")

#     # def test_2_analyze_functions(self):
#     #     self.assertEqual(len(self._function_pool._function_nodes), 0)

#     #     analyze_functions = AnalyzeFunctions("functions")
#     #     analyze_functions.context.load_from_file(
#     #         self._path + "/analyze_constraints.pkl"
#     #     )

#     #     asyncio.run(analyze_functions.run())
#     #     self.assertGreater(len(self._constraint_pool._constraint_nodes), 0)
#     #     analyze_functions.context.save_to_file(f"{self._path}/analyze_functions.pkl")

#     def test_3_design_functions(self):
#         design_functions = DesignFunctionAsync("design functions async")
#         design_functions.context.load_from_file(self._path + "/analyze_functions.pkl")

#         asyncio.run(design_functions.run())

#         design_functions.context.save_to_file(f"{self._path}/design_functions.pkl")

#     def test_4_write_functions(self):
#         write_function = WriteFunctionsAsync("design functions async")
#         write_function.context.load_from_file(self._path + "/design_functions.pkl")

#         asyncio.run(write_function.run())
#         write_function.context.save_to_file(f"{self._path}/write_functions.pkl")

#         for body in self._function_pool.functions_body:
#             self.assertIsNotNone(body)

#     def test_5_write_run(self):
#         write_run = WriteRun("write run")
#         write_run.context.load_from_file(self._path + "/write_functions.pkl")
#         # print([body for body in self._function_pool.functions_body])
#         asyncio.run(write_run.run())
#         write_run.context.save_to_file(f"{self._path}/write_run.pkl")

#     def test_6_code_review(self):
#         code_review = CodeReviewAsync("code review")
#         code_review.context.load_from_file(self._path + "/write_run.pkl")
#         asyncio.run(code_review.run())
#         code_review.context.save_to_file(f"{self._path}/code_review.pkl")


# if __name__ == "__main__":
#     suite = unittest.TestSuite()
#     # execute in such order
#     # suite.addTest(TestAction('test_1_analyze_constraint'))
#     # suite.addTest(TestAction("test_2_analyze_functions"))
#     # suite.addTest(TestAction("test_3_design_functions"))
#     # suite.addTest(TestAction('test_4_write_functions'))
#     # suite.addTest(TestAction('test_5_write_run'))
#     # suite.addTest(TestAction('test_6_code_review'))
#     # unittest.TextTestRunner().run(suite)
