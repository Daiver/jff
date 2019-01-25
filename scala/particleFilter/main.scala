
import util.Random.nextInt
import util.Random.nextFloat

object Pf{
    def getRandomVec(N: Int, max_value: Int) = {
        Seq.fill(N)(nextInt(max_value))
    }

    def mutate(field: Seq[Int], max_value: Int) = {
        def truncate(x: Int): Int = {
            if(x > max_value) x % max_value
            else if(x < 0) truncate(-x)
            else x
        }
        field.map( (x: Int) =>
            if(nextInt(100) < 30)
                truncate(x + nextInt(max_value) - max_value/2)
            else
                x
        )
    }

    def fitness(field: Seq[Int]) = {
        field.filter((x: Int) => x%2 == 0).sum
    }

    def particleFilter(particles_init: Seq[Seq[Int]], max_value: Int, iter_num: Int, max_iter: Int): Seq[Seq[Int]] = {
        val particles_par = particles_init.par.map((x: Seq[Int]) => mutate(x, max_value))
        val weights = particles_par.map(fitness)
        val particles = particles_par.toList

        val max_weight = weights.max
        var beta = 0.0
        var index = nextInt(particles.length)
        val new_particles = particles.map( (x: Seq[Int]) => {
            beta += nextFloat * 2.0 * max_weight
            //println(beta, index)
            while(beta > weights(index))
            {
                beta -= weights(index)
                index = (index + 1) % particles.length
            }
            particles(index)
        })
        if(iter_num == max_iter)
            new_particles
        else
            particleFilter(new_particles, max_value, iter_num+1, max_iter)
    }

    def main(args:Array[String]){
        val a = Seq.fill(5000)(getRandomVec(5, 10))
        //println(a)
        //println(a.map((x:Seq[Int]) => mutate(x, 100)))
        val b =(particleFilter(a, 10, 0, 5000))
        //println(b)
        println(b.maxBy(fitness))
    }
}

