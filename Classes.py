class Student:
	def __init__(self, name, attendance = "Absent"):
		self.name = name
		self.attendance = attendance

	def __str__(self) -> str:
		return self.name + ", " + self.attendance

class AttendanceList:
	def __init__(self, file):
		self.list = []
		for line in file:
			self.list.append(Student(line))

	def addStudent(self, student):
		self.list.append(student)

	def getList(self):
		return self.list