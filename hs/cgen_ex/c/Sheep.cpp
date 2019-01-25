#define CGEN_OUTPUT_INTERN
#include "Sheep.h"
void Animals_Sheep_delete(Sheep* this_ptr)
{
    delete this_ptr;
}

Sheep* Animals_Sheep_new(int wooliness_level_)
{
    return new Sheep(wooliness_level_);
}

void Animals_Sheep_make_sound(Sheep* this_ptr)
{
    this_ptr->make_sound();
}

void Animals_Sheep_shear(Sheep* this_ptr)
{
    this_ptr->shear();
}

