#ifndef CGEN_ANIMAL_H
#define CGEN_ANIMAL_H

#include <Sheep.h>
#include <Dog.h>
#include <Animal.h>

extern "C"
{

using namespace Animals;



#ifdef CGEN_HS
#endif

void Animals_Animal_delete(Animal* this_ptr);
void Animals_Animal_make_sound(Animal* this_ptr);
int Animals_Animal_get_age(const Animal* this_ptr);
void Animals_Animal_increment_age(Animal* this_ptr);

}

#endif

