package scheduler

import (
	"math"
	"math/rand"
	"sort"

	"github.com/docker/swarmkit/api"
	"github.com/docker/swarmkit/log"

	"github.com/skelterjohn/go.matrix"
)

const (
	MB = 1024 * 1024
	GB = 1024 * MB

	CORE = 1000 * 1000 * 1000

	MEM_REF = 16 * GB
	CPU_REF = 4 * CORE

	PLAN_SIZE = 2048
)

const (
	W_c = 0.5
	W_m = 0.5
)

type Config struct {
	Ants  int
	Q     float64
	Rho   float64
	Alpha float64
	Beta  float64
}

type Plan [PLAN_SIZE]int

func P(n NodeInfo) float64 {
	e := 0.0
	R_c := float64(n.AvailableResources.NanoCPUs) / float64(n.Description.Resources.NanoCPUs)
	R_m := float64(n.AvailableResources.MemoryBytes) / float64(n.Description.Resources.MemoryBytes)
	log.L.Debugf("R_c: %f, R_m: %f\n", R_c, R_m)
	return (W_c * (R_c + e)) + (W_m * (R_m + e))
}

func greedyInit(taskGroup map[string]*api.Task, nodes []NodeInfo) *matrix.DenseMatrix {
	keys := []string{}
	for k := range taskGroup {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// row, col
	TAU := matrix.Zeros(len(taskGroup), len(nodes))
	// fill zero as resource value
	for i := 0; i < TAU.Rows(); i++ {
		for j := 0; j < TAU.Cols(); j++ {
			if TAU.Get(i, j) == 0.0 {
				TAU.Set(i, j, P(nodes[j]))
			}
		}
	}

	col := 0
	for row, k := range keys {
		task := taskGroup[k]
		node := nodes[col]

		// TODO: if assign OK
		// greedy should assign tasks and reduce the resource
		node.AvailableResources.MemoryBytes -= task.Spec.Resources.Reservations.MemoryBytes
		node.AvailableResources.NanoCPUs -= task.Spec.Resources.Reservations.NanoCPUs
		nodes[col] = node

		tau := P(node)
		TAU.Set(row, col, tau)

		// Simply RR over nodes
		col = (col + 1) % len(nodes)
	}

	return TAU
}

func taskFitsNode(task *api.Task, node NodeInfo) bool {
	t := task.Spec.Resources.Reservations
	n := node.AvailableResources
	return (n.MemoryBytes >= t.MemoryBytes) && (n.NanoCPUs >= t.NanoCPUs)
}

func applyPlan(plan []int, tasks map[string]*api.Task, nodes []NodeInfo) []NodeInfo {
	keys := []string{}
	for k := range tasks {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	plan_idx := 0
	for _, k := range keys {
		node_idx := plan[plan_idx]
		node := nodes[node_idx]
		task := tasks[k]

		node.AvailableResources.MemoryBytes -= task.Spec.Resources.Reservations.MemoryBytes
		node.AvailableResources.NanoCPUs -= task.Spec.Resources.Reservations.NanoCPUs

		nodes[node_idx] = node
		plan_idx++
	}

	return nodes
}

func Optimize(taskGroup map[string]*api.Task, nodes []NodeInfo, config *Config) (float64, []int) {
	ANTS := config.Ants
	Q := config.Q
	RHO := config.Rho
	ALPHA := config.Alpha
	BETA := config.Beta

	refResources := make([]api.Resources, len(nodes))
	for i, node := range nodes {
		refResources[i] = api.Resources{
			MemoryBytes: node.AvailableResources.MemoryBytes,
			NanoCPUs:    node.AvailableResources.NanoCPUs,
		}
	}

	TAU_0 := greedyInit(taskGroup, nodes)
	TAU := TAU_0.Copy()

	taskNames := []string{}
	for name := range taskGroup {
		taskNames = append(taskNames, name)
	}
	sort.Strings(taskNames)

	plans := make(map[Plan]int)
	for i := 0; i < ANTS; i++ {
		var plan Plan
		log.L.Infof("== ANT: %d\n", i)

		// reset
		for n, refResource := range refResources {
			node := nodes[n]
			node.AvailableResources.MemoryBytes = refResource.MemoryBytes
			node.AvailableResources.NanoCPUs = refResource.NanoCPUs
			nodes[n] = node
		}

		// shuffle node orders
		nodeOrders := make([]int, len(nodes))
		for i := 0; i < len(nodes); i++ {
			nodeOrders[i] = i
		}
		shuffleInts(nodeOrders)
		log.L.Debugf("node order=%v\n", nodeOrders)

		// loop over tasks
		for task_idx, name := range taskNames {

			ph := make([]float64, len(nodes))
			ph_sum := 0.0

			task := taskGroup[name]

			// ant tries to put the task on to node
			for order := 0; order < len(nodeOrders); order++ {
				// pick order randomly
				node_idx := nodeOrders[order]

				node := nodes[node_idx]
				tau := TAU.Get(task_idx, node_idx)
				log.L.Debugf("tau=%f\n", tau)
				log.L.Debugf("node mem=%d\n", node.AvailableResources.MemoryBytes)
				log.L.Debugf("node cpu=%d\n", node.AvailableResources.NanoCPUs)

				if taskFitsNode(task, node) {
					node.AvailableResources.MemoryBytes -= task.Spec.Resources.Reservations.MemoryBytes
					node.AvailableResources.NanoCPUs -= task.Spec.Resources.Reservations.NanoCPUs
				} else {
					// TODO need to RR over nodes
					log.L.Debug("task not fit node")
					continue
				}

				nu := P(node)
				log.L.Debugf("nu=%f\n", nu)
				ph[node_idx] = math.Pow(tau, ALPHA) * math.Pow(nu, BETA)
				ph_sum += ph[node_idx]
				// log.L.Debugf("ph[%d]=%f\n", node_idx, ph[node_idx])
				// log.L.Debugf("ph_sum[%d]=%f\n", node_idx, ph_sum)
			}

			p := make([]float64, len(nodes))
			for node_idx := 0; node_idx < len(nodes); node_idx++ {
				p[node_idx] = (ph[node_idx] / ph_sum)
			}
			log.L.Debugf("ph = %v\n", ph)
			log.L.Debugf("p  = %v\n", p)

			j := -1
			if rand.Float64() >= Q {
				j = arg_max(p)
			} else {
				j = pick(p)
			}

			plan[task_idx] = j
			log.L.Debugf("j = %d\n", j)

			if j < 0 {
				// not fit
				// what to do?
			} else {
				TAU.Set(task_idx, j, P(nodes[j]))

				log.L.Info("vaporize pheromone")
				if task_idx < len(taskNames)-1 {
					for j := 0; j < TAU.Cols(); j++ {
						TAU.Set(task_idx+1, j, (1.0-RHO)*TAU.Get(task_idx, j))
					}
					TAU.Set(task_idx+1, j, (1.0-RHO)*P(nodes[j]))
				}

				log.L.Info("reduce the chosen resource")
				node := nodes[j]
				node.AvailableResources.MemoryBytes -= task.Spec.Resources.Reservations.MemoryBytes
				node.AvailableResources.NanoCPUs -= task.Spec.Resources.Reservations.NanoCPUs
				nodes[j] = node
			}

		} // task loop
		print(TAU)
		print(probTran(TAU))

		if _, exist := plans[plan]; exist {
			plans[plan]++
		} else {
			plans[plan] = 1
		}
	}

	max := 0
	var max_plan Plan
	for plan, count := range plans {
		// fmt.Printf("%v = %d\n", plan[0:len(taskGroup)], count)
		if count > max {
			max = count
			max_plan = plan
		}
	}

	// fmt.Printf("max  = %d\n", max)
	// fmt.Printf("plan = %v\n", max_plan[0:len(taskGroup)])
	confidence := float64(max) / float64(ANTS)

	return confidence, max_plan[0:len(taskGroup)]
}
