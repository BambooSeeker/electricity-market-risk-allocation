import unittest
from itertools import combinations
import numpy as np
from risk_allocation import boundary, fixed_count_exchange, exchange_frontier, value_frontier


class AllocationTests(unittest.TestCase):
    def test_boundary_exact_by_enumeration(self):
        base=np.array([9,8,7,6,5,4,3,2,1,0.])
        fill=np.array([0,0,1,5,8,7,4,3,2,1.])
        mask,core=boundary(base,fill,.4,.2,.7)
        eligible=[2,3,4,5,6]
        best=max(sum(fill[list(c)]) for c in combinations(eligible,2))
        self.assertAlmostEqual(fill[mask].sum(),best)
        self.assertEqual(mask.sum(),4)
        self.assertTrue(mask[core].all())

    def test_frontier_exact_and_nondominated(self):
        removes=[.7,.2,.5]; adds=[.8,.9,.1]
        gain,efficient=exchange_frontier(removes,adds,3)
        for count in range(4):
            brute=max(sum(a)-sum(r) for a in combinations(adds,count) for r in combinations(removes,count))
            self.assertAlmostEqual(gain[count],brute)
        self.assertEqual(efficient,[0,1,2])

    def test_fixed_count_policy_invariants(self):
        base=np.arange(20,0,-1,dtype=float)
        start,core=boundary(base,base,.4,.2,.6)
        result,audit=fixed_count_exchange(base,-base,start,core,np.ones(20,bool),.8,.25)
        self.assertEqual(result.sum(),start.sum())
        self.assertTrue(result[core].all())
        self.assertEqual((result!=start).sum(),2*audit['count'])
        self.assertLessEqual(audit['count'],audit['cap'])

    def test_economic_30_percent_condition(self):
        frame=value_frontier([0,.1,.2,.3,.4],[0,788,1576,2364,3152],[0,31951.9,48833.3,59647.7,63222.9])
        self.assertAlmostEqual(frame.iloc[2].normalized_break_even_cost,6.86193,places=4)
        self.assertAlmostEqual(frame.iloc[3].normalized_break_even_cost,2.26853,places=4)
        values=.5*np.array([0,31951.9,48833.3,59647.7,63222.9])-4*np.array([0,788,1576,2364,3152])
        self.assertEqual(int(values.argmax()),3)

    def test_no_eligible_addition(self):
        base=np.arange(20,0,-1,dtype=float)
        start,core=boundary(base,base,.4,.2,.6)
        result,audit=fixed_count_exchange(base,base,start,core,np.zeros(20,bool),.8,.25)
        np.testing.assert_array_equal(result,start)
        self.assertEqual(audit['count'],0)

    def test_zero_cap_and_ties(self):
        base=np.ones(20)
        start,core=boundary(base,base,.4,.2,.6)
        result,audit=fixed_count_exchange(base,base,start,core,np.ones(20,bool),.8,0)
        np.testing.assert_array_equal(result,start)
        self.assertEqual(audit['count'],0)
        self.assertEqual(np.flatnonzero(start).tolist(),list(range(8)))

    def test_negative_gain_endpoint_is_dominated(self):
        gain,efficient=exchange_frontier([.8],[.2],1)
        self.assertLess(gain[1],0)
        self.assertEqual(efficient,[0])


if __name__=='__main__':
    unittest.main()
