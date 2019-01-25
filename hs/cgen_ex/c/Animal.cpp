#define CGEN_OUTPUT_INTERN
#include "Animal.h"
void Animals_Animal_delete(Animal* this_ptr)
{
    delete this_ptr;
}

void Animals_Animal_make_sound(Animal* this_ptr)
{
    this_ptr->make_sound();
}

int Animals_Animal_get_age(const Animal* this_ptr)
{
    return this_ptr->get_age();
}

void Animals_Animal_increment_age(Animal* this_ptr)
{
    this_ptr->increment_age();
}

