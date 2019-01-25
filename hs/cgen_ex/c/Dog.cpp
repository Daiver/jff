#define CGEN_OUTPUT_INTERN
#include "Dog.h"
void Animals_Dog_delete(Dog* this_ptr)
{
    delete this_ptr;
}

Dog* Animals_Dog_new()
{
    return new Dog();
}

void Animals_Dog_make_sound(Dog* this_ptr)
{
    this_ptr->make_sound();
}

