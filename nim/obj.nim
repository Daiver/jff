type
  Person = ref object of RootObj
    name*: string  # эта * означает, что `name` будет доступно из других модулей
    age: int       # а это поле будет недоступно из других модулей

  Student = ref object of Person # Student унаследован от Person
    id: int                      # с дополнительным полем id

var
  student: Student
  person: Person
assert(student of Student) # вернёт true
# конструируем объект:
student = Student(name: "Anton", age: 5, id: 2)
echo student[]
