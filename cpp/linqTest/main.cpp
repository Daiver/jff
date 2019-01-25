#include <stdio.h>
#include "linq.h"

struct student_t
{
    std::string last_name;
    std::vector<int> scores;
};

int main()
{

	std::vector<student_t> students = {
		{"Omelchenko", {97, 72, 81, 60}},
		{"O'Donnell", {75, 84, 91, 39}},
		{"Mortensen", {88, 94, 65, 85}},
		{"Garcia", {97, 89, 85, 82}},
		{"Beebe", {35, 72, 91, 70}} 
	};

	auto scores = LINQ(from(student, students) 
					   from(score, student.scores) 
					   where(score > 86) 
					   select(std::make_pair(student.last_name, score)));

	for (auto x : scores){
		printf("%s score: %i\n", x.first.c_str(), x.second);
	}

    return 0;
}
